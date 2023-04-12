from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen

from src.conf import VERBOSE


class TrainScreen(Screen):
    logs = StringProperty()

    def __init__(self, **kwargs):
        self.logs = "Waiting to Starting training ..."
        super(TrainScreen, self).__init__(**kwargs)

    def disconnect(self):
        self.manager.client.disconnect()
        self.manager.current = 'welcome'

    def log(self, typ, txt, end=""):
        entry = None
        if typ == "info":
            entry = f"[color=FFFFFF]{txt}[/color]"
        elif typ == "warning":
            entry = f"[color=00FF00]{txt}[/color]"
        elif typ == "success":
            entry = f"[color=808000]{txt}[/color]"
        elif typ == "result":
            entry = f"[color=1E90FF]{txt}[/color]"
        elif typ == "error":
            entry = f"[color=FF0000]{txt}[/color]"
        else:
            if VERBOSE > 0:
                entry = f"[color=C0C0C0]{txt}[/color]"
        if entry:
            if end == "\r":
                self.logs = self.logs[:self.logs.rfind('\n')]
            self.logs = '\n'.join([self.logs, entry])
        return self.logs
