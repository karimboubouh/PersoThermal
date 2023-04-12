import os

import numpy as np
from kivy import platform
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.screenmanager import Screen
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager

from src import protocol
from src.conf import BRIDGE_HOST, BRIDGE_PORT


class ConfScreen(Screen):
    request_data = BooleanProperty()
    connect_logs = StringProperty()
    ds_path = StringProperty()
    connect_btn = StringProperty()
    join_disabled = BooleanProperty()

    def __init__(self, **kwargs):
        super(ConfScreen, self).__init__(**kwargs)
        self.dialog = None
        self.connect_logs = "Connect to get the list of your neighbors within the network"
        self.bridge_host = BRIDGE_HOST
        self.bridge_port = BRIDGE_PORT
        self.request_data = True
        self.ds_path = ""
        self.connect_btn = "Connect"
        self.join_disabled = True
        self.share_logs = True
        self.dataset_path = ""
        Clock.schedule_once(self.init, 1)
        self.file_manager = MDFileManager(
            select_path=self.select_path,
            # preview=True,
        )

    def on_request_data(self, checkbox, value):
        if value:
            # print('The checkbox', checkbox, 'is active', 'and', value, 'state')
            self.ids.ds_path = ""
            self.ids.ds_label.size_hint_y = None
            self.ids.ds_label.size_hint_x = None
            self.ids.ds_label.text = ""
            self.request_data = True
        else:
            self.ds_path = f"Select your dataset folder or file"
            self.ids.ds_label.size_hint_y = 1
            self.ids.ds_label.size_hint_x = 1
            self.ids.ds_label.text = self.ds_path
            self.request_data = False

    def init(self, *args):
        self.ids.bridge_host.text = self.bridge_host
        self.ids.bridge_host.focus = True
        self.ids.bridge_host.focus = False
        self.ids.bridge_port.text = str(self.bridge_port)
        self.ids.bridge_port.focus = True
        self.ids.bridge_port.focus = False
        self.ids.request_data.active = self.request_data
        self.ids.share_logs.active = self.share_logs

    def file_manager_open(self):
        path_root = '/storage/emulated/0/' if platform == 'android' else '/'
        self.file_manager.show(path_root)

    def select_path(self, path):
        self.file_manager.close()
        self.dataset_path = os.path.dirname(path)
        self.ids.ds_label.text = f"Selected dataset: {self.dataset_path}"
        self.manager.node.dataset_path = self.dataset_path
        toast(path)

    def connect(self):
        host = self.ids.bridge_host.text
        port = int(self.ids.bridge_port.text)
        self.connect_logs = f"HOST:{self.manager.node.host} / PORT:{self.manager.node.port}\n"
        self.dialog = MDDialog(title="Connection ...")  # , auto_dismiss=False
        try:
            if self.manager.node.connect_bridge(host, port):
                self.ids.connect_btn.disabled = True
                self.connect_btn = "Connected"
                toast("Connected successfully")
                self.manager.node.bridge.send_pref(self.request_data, self.share_logs)
                self.connect_logs = "Connecting to a set of peers ..."
            else:
                self.dialog.text = f"Could not connect to bridge."
                self.dialog.open()
                return False
        except Exception as e:
            self.dialog.text = f"Error while connecting to bridge: {str(e)}"
            self.dialog.open()
            return False

    def toggle_join(self, *args):
        self.ids.join_btn.disabled = not self.ids.join_btn.disabled
        self.manager.node.bridge.send(protocol.return_method("populate", {'s': True}))

    def log_pref(self):
        pref = ""
        pref = "\n".join([pref, f"[b]Id[/b] {self.manager.node.id}"])
        pref = "\n".join([pref, f"[b]Neighbors[/b] {self.manager.node.neighbors_ids}"])
        pref = "\n".join([pref, f"[b]Dataset[/b] {len(self.manager.node.dataset['Y_train'])} train samples"])
        pref = " | ".join([pref, f"{len(self.manager.node.dataset['Y_test'])} test samples"])
        pref = "\n".join([pref, f"[b]Epochs[/b] {self.manager.node.params.epochs} epochs."])
        pref = "\n".join([pref, f"[b]Batch size[/b] {self.manager.node.params.batch_size} samples."])
        self.connect_logs = pref
        Clock.schedule_once(self.toggle_join, 0)

    def join_train(self):
        self.manager.current = 'train'
