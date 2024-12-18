from .ac_mo import ACCriticMO
from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .ac_ns_mo import ACCriticNSMo


REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_mo"] = ACCriticMO
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["ac_critic_ns_mo"] = ACCriticNSMo


def register_pac_critics():
    from .pac_ac import PACCritic
    from .pac_ac_ns import PACCriticNS
    from .pac_dcg_ns import DCGCriticNS

    REGISTRY["pac_critic"] = PACCritic
    REGISTRY["pac_critic_ns"] = PACCriticNS
    REGISTRY["pac_dcg_critic_ns"] = DCGCriticNS
