from kivy.uix.screenmanager import ScreenManager

from src.node import Node


class ScreenManagement(ScreenManager):
    def __init__(self, **kwargs):
        super(ScreenManagement, self).__init__(**kwargs)
        self.node = Node(self)
