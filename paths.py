import os
from pathlib import Path
from typing import List

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
# RAW_DATASETS_PATH = HOME_PATH + "../Datasets/"
DATA_PATH = HOME_PATH + "../CachedDatasets/" # You set this

def open_token(name):
    with open(name + ".token", "r") as f:
        token = f.read()
    return token

def mkdir(path: str):
    if "." in path:
        Path(path[:path.rfind("/") + 1]).mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

def split_extension(file_path: str):
    if "." in file_path:
        index = file_path.rfind(".")
        return file_path[:index], file_path[index:]
        # Path(path[:path.rfind("/") + 1]).mkdir(parents=True, exist_ok=True)


class DirectoryPathManager:
    def __init__(self, base_path, base_unit_is_file, max_files_per_subfolder=-1, max_depth=None):
        self.base_path = base_path
        if base_unit_is_file:
            self.file_paths = get_all_files_in_directory(base_path, max_depth=max_depth,
                                                         max_files_per_subfolder=max_files_per_subfolder)
        else:
            self.file_paths = get_all_final_folders_in_directory(base_path, max_depth=max_depth)

    def get_files_absolute(self) -> List[str]:
        return [file.as_absolute() for file in self.file_paths]

    def get_files_relative(self, extension=True) -> List[str]:
        return [file.as_relative(extension) for file in self.file_paths]

    def get_file_names(self, extension=True):
        if extension:
            return [file.file_name + file.extension for file in self.file_paths]
        else:
            return [file.file_name for file in self.file_paths]

    @staticmethod
    def mkdir(path):
        mkdir(path)

    @staticmethod
    def split_extension(file_path):
        return split_extension(file_path)

class FileSplitter:
    def __init__(self, path):
        extension_index = path.find(".")
        file_name_index = path.rfind("/")
        if extension_index == -1:
            self.is_file = False
        else:
            self.is_file = True
            self.extension = path[extension_index:]
            self.file_name = path[file_name_index+1:extension_index]
        self.directories = path[:file_name_index].split("/")
        self.num_directories = len(self.directories)

class FilePath:
    def __init__(self, base_path, subfolder_path, file_name, is_folder):
        if is_folder:
            self.extension = "/"
            self.file_name = file_name
        else:
            self.extension = file_name[file_name.find("."):]
            self.file_name = file_name[:file_name.find(".")]
        self.subfolder_path = subfolder_path
        self.base_path = base_path
        if len(subfolder_path) == 0:
            self.subfolder_split = []
        else:
            self.subfolder_split = self.subfolder_path[:-1].split("/")

        # self.subfolder_split = [folder if len(folder) > 0 for folder in self.subfolder_split]

    def as_subfolder_string(self):
        name = ""
        for subfolder in self.subfolder_split:
            name += subfolder + "_"
        name += self.file_name
        return name

    def as_absolute(self):
        return self.base_path + self.subfolder_path + self.file_name + self.extension

    def as_relative(self, extension=True):
        if extension:
            return self.subfolder_path + self.file_name + self.extension
        else:
            return self.subfolder_path + self.file_name

def folder_contains_no_folders(folder_path_absolute):
    contents = os.listdir(folder_path_absolute)
    for content in contents:
        if os.path.isdir(folder_path_absolute + content):
            return False
    return True

def get_all_final_folders_in_directory(base_path, subfolder_path="", max_depth=None):
    base_contents = os.listdir(base_path + subfolder_path)
    base_contents.sort()

    folder_paths = []
    for content in base_contents:
        if not os.path.isdir(base_path + subfolder_path + content):
            continue
        if folder_contains_no_folders(base_path + subfolder_path + content + "/"):
            folder_paths.append(FilePath(base_path, subfolder_path, content, is_folder=True))
        
        else:
            if max_depth is not None and max_depth <= 0:
                continue
            folder_paths.extend(get_all_final_folders_in_directory(base_path, max_depth=max_depth-1 if max_depth is not None else None,
                                                          subfolder_path=subfolder_path + content + "/"))
    return folder_paths

def get_all_files_in_directory(base_path, subfolder_path="", max_files_per_subfolder=-1, max_depth=None) -> List[FilePath]:
    base_contents = os.listdir(base_path + subfolder_path)
    base_contents.sort()
    if max_files_per_subfolder > 0:
        max_files = len(base_contents)
        if max_files_per_subfolder < max_files:
            max_files = max_files_per_subfolder
        base_contents = base_contents[:max_files]
    files_paths = []

    for content in base_contents:
        if os.path.isfile(base_path + subfolder_path + content):
            files_paths.append(FilePath(base_path, subfolder_path, content, is_folder=False))
        else:
            if max_depth is not None and max_depth <= 0:
                continue
            files_paths.extend(get_all_files_in_directory(base_path, max_depth=max_depth-1 if max_depth is not None else None,
                                                          subfolder_path=subfolder_path + content + "/",
                                                          max_files_per_subfolder=max_files_per_subfolder))
    return files_paths

