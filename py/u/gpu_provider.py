import threading
from multiprocessing import Lock, Queue

from u.config import AVAILABLE_GPUS, MAIN_THREAD, SET_PARALLELISM
from u.gpu import Gpu
from u.log import Log


class GpuProvider:
    _available_gpus = Queue()
    for i in AVAILABLE_GPUS:
        _available_gpus.put(i)
    gpu_count = len(AVAILABLE_GPUS)
    lock = Lock()

    @staticmethod
    def get_gpu(process_name):
        try:
            SET_PARALLELISM()
            GpuProvider.lock.acquire()
            threading.current_thread().name = process_name
            gpu_id = GpuProvider._available_gpus.get()
            Log.info_main("%s assigned %s/%s", process_name, gpu_id, GpuProvider.gpu_count)
            return Gpu(gpu_id)
        finally:
            GpuProvider.lock.release()

    @staticmethod
    def return_gpu(gpu, process_name):
        try:
            GpuProvider.lock.acquire()
            GpuProvider._available_gpus.put(gpu.gpu_id)
            threading.current_thread().name = MAIN_THREAD
            Log.info_main("%s returned %s/%s", process_name, gpu.gpu_id, GpuProvider.gpu_count)
        finally:
            GpuProvider.lock.release()
