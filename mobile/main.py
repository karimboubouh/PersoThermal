from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivymd.app import MDApp
from src.screens import *

Window.size = (336, 600)
# Window.release_all_keyboards()


class ThermalApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.primary_hue = "700"
        return Builder.load_file('src/template.kv')


ThermalApp().run()

"""
--> Solution 1
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('192.168.0.1', 0))
    s.connect(('...'))
--> Solution 2
    # from socket.h
    # define SO_BINDTODEVICE 25
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, 25, 'eth0')
"""