from .agentprm import audit_agentprm
from .contractbench import audit_contractbench
from .taubench import audit_taubench
from .toolace import audit_toolace

ADAPTERS = {
    "agentprm": audit_agentprm,
    "contractbench": audit_contractbench,
    "toolace": audit_toolace,
    "taubench": audit_taubench,
}
