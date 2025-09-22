import multiprocessing
from multiprocessing import Pool, Queue, Process
from tqdm import tqdm
import queue

SENTINEL_VALUE = None
ERROR_STATUS = "Proc Error: 123047"

def pbar_queue_task(pbar_queue, pbar, num_workers, timeout=None):
    num_errors = 0
    sentinels_received = 0
    
    while sentinels_received < num_workers:  # You'd need to pass num_workers
        try:
            item = pbar_queue.get(timeout=timeout)
            if item is SENTINEL_VALUE:
                sentinels_received += 1
                continue

            else:
                process_item, error_status = item
                if error_status == ERROR_STATUS:
                    num_errors += 1
                pbar.update()

                if isinstance(process_item, tuple):
                    pbar_index_name = process_item[0]
                else:
                    pbar_index_name = process_item

                pbar.set_description(f"Processing {pbar_index_name} Errors: {num_errors} Finished: {sentinels_received}/{num_workers}")
        except queue.Empty:
            print("Progress bar timed out waiting for updates")
            break


class MultiProcQueueProcessing:
    def __init__(self, args_global, task, num_workers):
        # task:
        # f(process_item, *args_global):
        #     if error: return ERROR_STATUS
        #     else: return None or desired output
        # by convention, the process_item is a string or a number. A more complex item can be extracted from args_global
        # execute the useful task to parallelize. Takes in the item to be processed and the constant global arguments.
        # args_global: a tuple of global arguments (garg1, garg2, ...)
        self.args_global = args_global
        self.task = task
        self.num_workers = num_workers

    def _queue_task(self, queue, proc_num, outputs, args_global, pbar_queue: Queue):
        task_outputs = []
        try:
            while True:
                # Grab an item
                try:
                    # Add timeout to prevent hanging on queue.get()
                    args_item = queue.get(timeout=5)
                except queue.Empty:
                    print(f"Worker {proc_num}: No more items in queue")
                    break

                # If sentinel, stop and signal pbar
                if args_item is SENTINEL_VALUE:
                    pbar_queue.put(SENTINEL_VALUE)
                    outputs[proc_num] = task_outputs
                    break
                # Otherwise, process
                else:
                    try:
                        status = self.task(*((args_item, ) + args_global))
                        if status != ERROR_STATUS:
                            task_outputs.append(status)
                    except Exception as e:
                        print("task error:", e)
                        continue

                    # This signals the progress bar
                    pbar_queue.put((args_item, status))
        except Exception as e:
            print("=" * 50)
            print(f"Worker {proc_num} crashed: {e}")
            # If crash, tell pbar that this is finished
            pbar_queue.put(SENTINEL_VALUE)  # Signal progress bar to continue
            outputs[proc_num] = task_outputs

    def process(self, process_list, timeout=None):
        # We create 2 sets of queues. One is for the mesh processing. The other is a surrogate for the tqdm bar (which would get copied across tasks otherwise)
        pbar = tqdm(process_list, smoothing=0.05)
        pbar_queue = Queue()

        # set up output
        manager = multiprocessing.Manager()
        outputs = manager.dict()

        # Set up queue for multiprocessing
        q = Queue()
        processes = []
        # Create processes
        for i in range(self.num_workers):
            p = Process(target=self._queue_task, args=(q, i, outputs, self.args_global, pbar_queue), name=f"Process-{i}")
            processes.append(p)
            p.start()
        p_pbar = Process(target=pbar_queue_task, args=(pbar_queue, pbar, self.num_workers, timeout), name=f"pbar")
        p_pbar.start()

        # Create queue to parse through
        for i in process_list:
            q.put(i)

        # Sentinel values: Tells workers when task is finished
        for i in range(self.num_workers):
            q.put(SENTINEL_VALUE)

        for p in processes:
            p.join(timeout=timeout)

        p_pbar.join(timeout=30)  # Wait max 30 seconds

        # print(outputs)
        outputs = [ent for sublist in outputs.values() for ent in sublist]
        return outputs

if __name__=="__main__":
    process_list = [1, 2, 3, 4, 5, 6]

    scale = 12
    center = 1

    num_workers = 3

    def task(x, scale, center):
        # print(x * scale + center)
        return (x * scale + center)

    processor = MultiProcQueueProcessing(args_global=(scale, center), task=task, num_workers=num_workers)
    output = processor.process(process_list)
    print(output)
