import json
import threading
from time import sleep

from prettytable import PrettyTable

class Dashboard:

    def __init__(self):
        self._watch_dash_board = False

    def start_printing_dashboard(self, source_function):
        self._watch_dash_board = True
        threading.Thread(target=self._print_stages_stats, args=[source_function]).start()

    def stop_printing_dashboard(self):
        self._watch_dash_board = False


    def _print_stages_stats(self, source_function):
        while self._watch_dash_board:
            data = source_function()
            table = PrettyTable()
            table.field_names = ["Stage number", "Address", "Tasks in work"]
            for stageNumber in data.keys():
                for addr in data[stageNumber].keys():
                    table.add_row([stageNumber, addr, data[stageNumber][addr]["load"]])

            print(table)
            sleep(3.0)


def get_test_data():
    with open('test_data.json', 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.start_printing_dashboard(get_test_data)
    sleep(20.0)
    dashboard.stop_printing_dashboard()

