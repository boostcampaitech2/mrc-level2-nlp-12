import os
import json


class DPRDataset:
    def __init__(self, wiki_data_path):
        self.wiki_data_path = wiki_data_path
        self.wiki = None

    def load_wiki_data(self, path=None):
        with open(path, "r") as f:
            self.wiki = json.load(f)
