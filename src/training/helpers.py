import pickle
from prefect import task


def load_pickle(path: str):
    with open(path, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: dict):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

@task(name="Load", tags=['Serialize'])
def task_load_pickle(path: str):
    return load_pickle(path)


@task(name="Save", tags=['Serialize'])
def task_save_pickle(path: str, obj: dict):
    save_pickle(path, obj)