import logging
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
from dataclasses import dataclass

from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress

logger = logging.getLogger(__name__)


@dataclass
class GUIPacket:
    class T:
        NEW_DESIGN = 0
        NEW_SPEC = 1
        NEW_PROOF = 2

    t: T
    iden: str = ""
    dname: str = ""
    sname: str = ""
    file: str = ""
    params: str = ""
    result: str = ""


class PYCGUI:
    def __init__(self) -> None:
        pass

    def push_update(self, data: GUIPacket):
        pass

    def update_progress(self, desc: str, progress: float):
        pass

    def reset_progress(self):
        pass

    def run(self, debug=True):
        pass


class WebGUI(PYCGUI):
    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "secret!"
        self.socketio = SocketIO(self.app)
        self._register_routes()

    def _register_routes(self):
        """Define Flask routes for the web interface."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

    # This function is part of the "rest of your app" and is called whenever new data is ready.
    def push_update(self, data: GUIPacket):
        # Here, data is a dictionary containing the update.
        # You can call this function from anywhere in your application.
        self.socketio.emit("data_update", {"message": data.message})
        logger.debug(f"Update sent: {data.message}")

    def run(self, debug=True):
        """
        Run the Flask-SocketIO server in its own thread so it doesn't block
        the main application logic.
        """

        def run_server():
            self.socketio.run(self.app, debug=debug, use_reloader=False)

        gui_thread = threading.Thread(target=run_server)
        gui_thread.daemon = True  # Allows the process to exit even if gui_thread runs.
        gui_thread.start()
        logger.debug("GUI is running in a background thread.")

    def update_progress(self, desc: str, progress: int):
        pass

    def reset_progress(self):
        pass


class RichGUI(PYCGUI):
    def __init__(self) -> None:
        self.window = self.mk_window()
        self.layout = self.window.renderable
        self.design_table = self.layout["designs"].renderable
        self.spec_table = self.layout["specs"].renderable
        self.proofs_table = self.layout["proofs"].renderable

        self.progress: Progress = self.layout["progress"].renderable
        self.bar_task = self.progress.add_task("bar", total=1)

    def push_update(self, data: GUIPacket):
        # Add the new data to the table
        if data.t == GUIPacket.T.NEW_DESIGN:
            self.design_table.add_row(data.dname, data.file)
        elif data.t == GUIPacket.T.NEW_SPEC:
            self.spec_table.add_row(data.sname, data.file, data.params)
        elif data.t == GUIPacket.T.NEW_PROOF:
            self.proofs_table.add_row(data.iden, data.sname, data.dname, data.result)

    def update_progress(self, desc: str, progress: int):
        self.progress.update(self.bar_task, description=desc, advance=progress)

    def reset_progress(self):
        self.progress.reset(self.bar_task, description="Ongoing task")

    def mk_window(self):
        layout = Layout()
        layout.split_column(
            Layout(name="designs", ratio=1),
            Layout(name="specs", ratio=1),
            Layout(name="proofs", ratio=1),
            Layout(name="progress", size=1),
        )

        design_table = Table(title="RTL Designs", expand=True)
        design_table.add_column("Design Name", style="cyan")
        design_table.add_column("File", style="cyan")
        layout["designs"].update(design_table)

        spec_table = Table(title="Specifications", expand=True)
        spec_table.add_column("Name", style="cyan")
        spec_table.add_column("File", style="cyan")
        spec_table.add_column("Params", style="cyan")
        layout["specs"].update(spec_table)

        proofs_table = Table(title="Proofs", expand=True)
        proofs_table.add_column("ID", style="cyan")
        proofs_table.add_column("Design", style="cyan")
        proofs_table.add_column("Spec", style="cyan")
        proofs_table.add_column("Result", style="cyan")
        layout["proofs"].update(proofs_table)

        progress = Progress()
        layout["progress"].update(progress)

        return Panel(layout, title="PyCaliper Status", expand=False, height=30)

    def run(self, debug=True):
        """
        Run the Flask-SocketIO server in its own thread so it doesn't block
        the main application logic.
        """

        def run_gui():
            with Live(self.window, refresh_per_second=4) as live:
                while True:
                    pass

        gui_thread = threading.Thread(target=run_gui)
        gui_thread.daemon = True  # Allows the process to exit even if gui_thread runs.
        gui_thread.start()
        logger.debug("GUI is running in a background thread.")
