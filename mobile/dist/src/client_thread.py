import pickle
import socket
import traceback
from threading import Thread
from struct import pack, unpack

import numpy as np
import plyer
from kivymd.toast import toast

from . import message
from .models import LogisticRegression
from .utils import adaptive_batch_size, input_size, mah


class ClientThread(Thread):

    def __init__(self, client):
        super(ClientThread, self).__init__()
        self.client = client
        self.sock = client.sock
        self.terminate = False

    def run(self):
        # Wait for messages from server
        while not self.terminate:
            try:
                (length,) = unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(4096 if to_read > 4096 else to_read)

                if buffer:
                    data = pickle.loads(buffer)
                    if data and data['mtype'] == message.TRAIN_JOIN:
                        self.join_train(data['data'])
                    elif data and data['mtype'] == message.TRAIN_START:
                        self.client.local_train(data['data'])
                    elif data and data['mtype'] == message.TRAIN_STOP:
                        self.stop_train(data['data'])
                    elif data and data['mtype'] == message.DISCONNECT:
                        self.stop()
                    else:
                        toast(f"Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                toast(f"Corrupted message: {e}")
                traceback.print_exc()
            except socket.timeout:
                pass
            except Exception as e:
                self.terminate = True
                toast(f"Socket Exception: {e}")
                traceback.print_exc()

        self.sock.close()
        toast(f"Client disconnected")

    def send(self, msg):
        try:
            length = pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            toast(f"Socket error\n{e}")
        except Exception as e:
            toast(f"Exception\n{e}")

    def stop(self):
        self.terminate = True

    def join_train(self, data):
        self.client.params.lr = data.get('lr', self.client.params.lr)
        self.client.model = data.get('model', None)
        if data['model_name'] == "LR":
            features = input_size(data['model_name'], "mnist")
            self.client.model = LogisticRegression(features, lr=self.client.params.lr)
            self.client.model.batch_size = adaptive_batch_size(self.client.profile)
            self.client.log(log="Joined training, waiting to start ...")
        else:
            toast(f"Model {data['model_name']} not supported.")
            exit(0)

    def stop_train(self, data):
        # go to predict screen
        self.client.log(log="Training finished.")
        self.client.manager.current = 'predict'
        battery_usage = mah(self.client.battery_start, self.client.battery_capacity)
        summary = f"Training finished.\nAccuracy: {data['performance'][1]}\nLoss: {data['performance'][0]}\n" \
                  f"Local battery usage: {round(battery_usage, 4)}mAh\n" \
                  f"Global battery usage: {round(data['battery_usage'], 4)}mAh\n" \
                  f"Local iteration cost: {round(float(np.mean(self.client.iteration_cost)), 4)}s\n" \
                  f"Global iteration cost: {round(data['iteration_cost'], 4)}s"
        self.client.manager.get_screen("predict").ids.train_summary.text = summary
