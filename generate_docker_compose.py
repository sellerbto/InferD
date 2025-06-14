import yaml
import ipaddress

COMPOSE_FILENAME = "docker-compose.generated.yml"
INFERD_CONFIG    = "petals/inferd.yaml"
NETWORK_SUBNET   = "172.28.0.0/16"
IP_START         = "172.28.0.2"

with open(INFERD_CONFIG, 'r') as f:
    cfg = yaml.safe_load(f)

stages = cfg["stages"]

nodes = cfg['nodes']
num_nodes = len(nodes)



net   = ipaddress.ip_network(NETWORK_SUBNET)
start = ipaddress.IPv4Address(IP_START)
ips   = [str(start + i) for i in range(num_nodes)]

compose = {
    "version": "3.8",
    "services": {},
    "networks": {
        "infernet": {
            "driver": "bridge",
            "ipam": {
                "config": [{"subnet": NETWORK_SUBNET}]
            }
        }
    }
}

bootstrap_addrs = [f"{ip}:7050" for ip in ips]

for idx, st in enumerate(nodes):
    svc_name       = st["name"]
    initial_stage  = st["stage"]
    host_7050      = str(7050 + idx)
    host_6050      = str(6050 + idx)
    pth_dir        = svc_name

    compose["services"][svc_name] = {
        "build": {
            "context": ".",
            "dockerfile": "Dockerfile",
            "args": {
                "PTH_DIR": pth_dir
            }
        },
        "container_name": svc_name,
        "ports": [
            f"{host_7050}:7050",
            f"{host_6050}:6050",
        ],
        "environment": [
            f"INITIAL_STAGE={initial_stage}",
            f"BOOTSTRAP_NODES={','.join(bootstrap_addrs)}",
            f"NODE_NAME={svc_name}"
        ],
        "networks": {
            "infernet": {"ipv4_address": ips[idx]}
        },
        "command": ["run_node.py"]
    }


# compose["services"]["test_path_finding"] = {
#     "build": {
#         "context": ".",
#         "dockerfile": "Dockerfile",
#         "args": {
#             "PTH_DIR": "test"
#         }
#     },
#     "container_name": "node_test",
#     "ports": [
#         f"{7050 + num_stages}:7050",
#         f"{6050 + num_stages}:6050",
#     ],
#     "networks": {
#         "infernet": {"ipv4_address": str(start + num_stages)}
#     },
#     "environment": [
#         "NODE_NAME=node_test"
#     ],
#     "working_dir": "/inferd",
#     "command": ["uv", "run", "python", "test_path_finding.py"]
# }

with open(COMPOSE_FILENAME, 'w') as f:
    yaml.safe_dump(compose, f, sort_keys=False, default_flow_style=False)

print(f"Generated {COMPOSE_FILENAME}")
