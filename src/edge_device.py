import pickle
import socket
import struct
from threading import Thread

import numpy as np

from src import protocol
import src.conf as C
from src.helpers import Map
from src.utils import log, wait_until, create_tcp_socket, get_ip_address


def edge_devices(args, count=1, rand_ids=False):
    if count < 1:
        return None
    if C.ML_ENGINE not in ["N3", "NumPy"]:
        log('error', f"Mobile devices currently only support NumPy based ML")
        exit()
    if args.mp == 0:
        log('error', f"You need to use message passing when edge devices are involved")
        exit()
    launcher = Bridge(count, args, rand_ids=rand_ids)
    launcher.start()
    wait_until(launcher.bridged, C.LAUNCHER_TIMEOUT, 1)
    if len(launcher.bridges) == count:
        log('success', f"All edge devices joined successfully")
    elif len(launcher.bridges) == 0:
        log('error', f"No device joined in {C.LAUNCHER_TIMEOUT} seconds")
        launcher.stop()
        exit()
    else:
        log('error', f"Only {len(launcher.bridges)} devices joined after waiting for {C.LAUNCHER_TIMEOUT} seconds")
        exit()

    return launcher


class Bridge(Thread):

    def __init__(self, nb_devices: int, args: Map, rand_ids=False):
        super(Bridge, self).__init__()
        self.nb_devices = nb_devices
        if rand_ids:
            self.devices_ids = np.random.choice(range(args.num_users), nb_devices, replace=False).tolist()
        else:
            self.devices_ids = np.arange(nb_devices).tolist()
        self.ids: list = list(range(nb_devices))
        self.args = args
        self.terminate = False
        self.host = get_ip_address()  # "127.0.0.1" # get_ip_address()
        self.port = C.LAUNCHER_PORT
        self.bridges = []
        self.waiting = {}
        self._init_server()

    def run(self):
        log('info', f"{self}: Waiting for {self.nb_devices} devices to join ...")
        while not self.terminate:
            try:
                conn, address = self.sock.accept()
                if not self.terminate:
                    bridge = DeviceBridge(self, self.ids.pop(), conn, address)
                    bridge.start()
                    self.bridges.append(bridge)
                    log('info', f"New device({bridge.id}) <{address}> joined")
            except socket.timeout:
                pass
            except Exception as e:
                log('error', f"{self}: Node Exception\n{e}")

        for device in self.bridges:
            device.stop()
        self.sock.close()
        log('log', f"{self}: Stopped")

    def is_edge_device(self, i):
        return i in self.devices_ids

    def bridged(self):
        return len(self.bridges) == self.nb_devices

    def populate_device(self, i, model, data, ids, clustered, sim):
        log('info', f"Populating device {i} ...")
        bridge: DeviceBridge = self.get_bridge_by_id(i)
        assert bridge is not None
        args = {'epochs': self.args.epochs, 'batch_size': self.args.batch_size, 'lr': self.args.lr,
                'momentum': self.args.momentum, 'gar': self.args.gar, 'frac': self.args.frac}
        info = {'id': i, 'args': args, 'model': model, 'ids': ids, 'clustered': clustered, 'similarity': sim}
        if bridge.request_data:
            info['dataset'] = data
        else:
            info['num_users'] = self.args.num_users
            info['ds_duplicate'] = C.DATASET_DUPLICATE
        bridge.populate(info)

        # return DeviceBridge
        return bridge

    @staticmethod
    def send(bridge, msg):
        bridge.send(msg)

    def get_bridge_by_id(self, bid):
        for bridge in self.bridges:
            if bridge.id == bid:
                return bridge
        return None

    def broadcast(self, msg):
        for bridge in self.bridges:
            self.send(bridge, msg)

    def stop(self):
        self.terminate = True
        self.sock.close()

    def _init_server(self):
        self.sock = create_tcp_socket()
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(C.SOCK_TIMEOUT)
        self.sock.listen(C.TCP_SOCKET_SERVER_LISTEN)
        self.host = self.sock.getsockname()[0]

    def __repr__(self):
        return f"Bridge({self.host}, {self.port})"
        # return f"Bridge{self.sock.getsockname()}"

    def __str__(self):
        return f"Bridge({self.host}, {self.port})"


class DeviceBridge(Thread):
    def __init__(self, bridge, bid, sock, address):
        super(DeviceBridge, self).__init__()
        self.bridge = bridge
        self.request_data = None
        self.share_logs = None
        self.neighbors_ids = []
        self.neighbors = []
        self.sock = sock
        self.address = address
        self.host = None
        self.port = None
        self.id = bid
        self.terminate = False
        self.callbacks = {}
        self.inference = "TODO"
        self.current_exec = None
        self.params = Map({
            'logs': []
        })

    def run(self):
        # Wait for messages from device
        while not self.terminate:
            try:
                (length,) = struct.unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(4096 if to_read > 4096 else to_read)
                if buffer:
                    data = pickle.loads(buffer)
                    if data and data['mtype'] == protocol.PREFERENCES:
                        self.handle_pref(data['data'])
                    elif data and data['mtype'] == protocol.CALL_METHOD:
                        self.call_method(data['data'])
                    elif data and data['mtype'] == protocol.NEIGHBORS:
                        self.handle_neighbors(data['data'])
                    elif data and data['mtype'] == protocol.RETURN_METHOD:
                        self.callbacks[data['data']['method']] = data['data']['return']
                    elif data and data['mtype'] == protocol.DEVICE_LOGS:
                        self.handle_logs(data['data'])
                    elif data and data['mtype'] == protocol.DISCONNECT:
                        self.handle_disconnect()
                    else:
                        log('error', f"{self}: Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                log('error', f"{self}: Corrupted message : {e}")
            except socket.timeout:
                pass
            except struct.error:
                # log('warning', f"{self} struct.error: {e}")
                pass
            except Exception as e:
                self.terminate = True
                log('error', f"{self} Exception: {e}")
        self.sock.close()
        log('log', f"{self}: disconnected")

    def send(self, msg):
        try:
            if self.terminate:
                log('log', f"{self} tries to send on terminated")
            length = struct.pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            log('error', f"{self}: Socket error: {e}: ")
        except Exception as e:
            log('error', f"{self}: Exception\n{e}")

    def stop(self):
        self.terminate = True

    def populate(self, info):
        self.send(protocol.call_method("populate", info))
        done = wait_until(self.return_method, C.FUNC_TIMEOUT, 1, "populate")
        if done and self.callbacks['populate']['s']:
            del self.callbacks['populate']
            log('success', f"{self} populated successfully")
        elif done:
            log("error", f"Error populating {self}")
        else:
            log('warning', f"Calling populate() timeout  after {C.FUNC_TIMEOUT} seconds")

    def connect(self, neighbor):
        self.send(protocol.call_method("connect", neighbor.id, neighbor.host, neighbor.port))
        done = wait_until(self.return_method, C.FUNC_TIMEOUT, 1, "connect")
        if done and self.callbacks['connect']['s']:
            self.neighbors.append(neighbor.id)
            del self.callbacks["connect"]
            return True
        elif done:
            log("error", self.callbacks['connect']['m'])
            return False
        else:
            log('warning', f"Calling connect() timeout  after {C.FUNC_TIMEOUT} seconds")
            return False

    def fit(self, args, inference=True, one_batch=False):
        self.send(protocol.call_method("fit", args, inference, one_batch))
        done = wait_until(self.return_method, C.FUNC_TIMEOUT, 1, "fit")
        if done and self.callbacks['fit']['s']:
            history = self.callbacks['fit']['m']
            del self.callbacks['fit']
            # for i, h in enumerate(history):
            #     log('', f"Epoch [{i}], val_loss: {h['val_loss']:.4f}, val_acc: {h['val_acc']:.4f}")
            return history
        else:
            log('warning', f"Calling fit() timeout  after {C.FUNC_TIMEOUT} seconds")
            return None

    def execute(self, func, *args, **kwargs):
        method = f"execute.{func.__name__}"
        msg = protocol.call_method(method, *args, **kwargs)
        self.send(msg)
        self.current_exec = method
        # self.wait_method(method)

    def wait_method(self, method):
        done = wait_until(self.return_method, C.FUNC_TIMEOUT, 1, method)
        if not done:
            log('warning', f"Calling execute({method}) timeout  after {C.FUNC_TIMEOUT} seconds")

    def return_method(self, key):
        return key in self.callbacks

    def call_method(self, d):
        if d['method'] == "log_results":
            self.log_results(*d['args'], **d['kwargs'])

    def log_results(self, t_eval):
        self.params.logs.append(t_eval)

    def handle_pref(self, data):
        self.host = data['host']
        self.port = data['port']
        self.request_data = data['request_data']
        self.share_logs = data['share_logs']

    def handle_neighbors(self, data):
        self.neighbors = data['nbrs']

    @staticmethod
    def handle_logs(data):
        log(data['typ'], data['txt'])

    def handle_disconnect(self):
        if self in self.bridge.bridges:
            self.bridge.bridges.remove(self)
        self.terminate = True
        self.sock.close()

    #  Private methods --------------------------------------------------------

    def __repr__(self):
        return f"DeviceBridge({self.id})"

    def __str__(self):
        return f"DeviceBridge({self.id})"
