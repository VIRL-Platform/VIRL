from .agent_template import AgentTemplate


__all__ = {
    'AgentTemplate': AgentTemplate,
}


def build_agent(agent_cfg):
    if agent_cfg.NAME in __all__:
        return __all__[agent_cfg.NAME](agent_cfg)
    else:
        return __all__['AgentTemplate'](agent_cfg)
