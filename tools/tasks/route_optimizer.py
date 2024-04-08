from .task_template import TaskTemplate

from virl.config import cfg
from virl.actions.navigation import build_navigator


class RouteOptimizer(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args, pipeline_cfg=None, **kwargs):
        if pipeline_cfg is None:
            pipeline_cfg = cfg.PIPELINE

        # Step 1: Route planning
        start_position = agent.start_position
        self.navigator = build_navigator(
            pipeline_cfg.NAVIGATION, platform, messager, start_position,
            output_dir=self.output_dir, resume=args.resume
        )

        # Step 2: Start navigation
        step_counter = 1
        while True:
            is_finish, current_geocode, info_dict = self.navigator.navigate(info_dict={})

            if is_finish:
                break

            if step_counter % cfg.get('SAVE_INTERVAL', 1e8) == 0:
                self.save_results()

            step_counter += 1

        # Output
        # visualize trajectory
        self.navigator.show_trajectory_on_the_map()
