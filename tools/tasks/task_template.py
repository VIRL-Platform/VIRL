
class TaskTemplate(object):
    def __init__(self, output_dir, logger):
        self.output_dir = output_dir
        self.logger = logger
        self.memory = None
        self.navigator = None

    def run(self, **kwargs):
        raise NotImplementedError

    def save_results(self):
        if self.memory is not None:
            self.memory.save_memory()

        if self.navigator is not None:
            self.navigator.save_navigator()
