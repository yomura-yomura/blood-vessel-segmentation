import pathlib

from .typing import FilePath

_project_root_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent
_data_dir_path: pathlib.Path | None = None
_kaggle_dataset_dir_path: pathlib.Path | None = None


__all__ = ["get_project_root_path", "get_data_dir_path", "get_kaggle_dataset_dir_path", "set_pj_struct_paths"]


def get_project_root_path() -> pathlib.Path:
    return _project_root_path


def get_data_dir_path() -> pathlib.Path:
    if _data_dir_path is None:
        return get_project_root_path() / "data"
    else:
        return _data_dir_path


def get_kaggle_dataset_dir_path() -> pathlib.Path:
    if _kaggle_dataset_dir_path is None:
        return get_data_dir_path() / "blood-vessel-segmentation"
    else:
        return _kaggle_dataset_dir_path


def set_pj_struct_paths(
    project_root_path: FilePath | None = None,
    data_dir_path: FilePath | None = None,
    kaggle_dataset_dir_path: FilePath | None = None,
) -> None:
    global _project_root_path, _data_dir_path, _kaggle_dataset_dir_path

    if project_root_path is not None:
        print(f"project_root_path changed: {_project_root_path} -> {project_root_path}")
        _project_root_path = pathlib.Path(project_root_path)

    if data_dir_path is not None:
        print(f"data_dir_path changed: {_data_dir_path} -> {data_dir_path}")
        _data_dir_path = pathlib.Path(data_dir_path)

    if kaggle_dataset_dir_path is not None:
        print(f"kaggle_dataset_dir_path changed: {_kaggle_dataset_dir_path} -> {kaggle_dataset_dir_path}")
        _kaggle_dataset_dir_path = pathlib.Path(kaggle_dataset_dir_path)
