def parse_ip_port(ip_port):
    ip, port =  ip_port.split(':') # ip
    port = int(port)
    return ip, port


async def min_max_load_stage(node_info, dht_map):
    lmin = float('inf')
    lmax = float('-inf')
    smin = None
    smax = None
    for stage_str, peers in dht_map.items():
        stage_i = int(stage_str)
        L = sum(p['load'] for p in peers.values()) if peers else 10**21

        if L < lmin or (L == lmin and stage_i < smin):
            lmin, smin = L, stage_i
        if L > lmax or (L == lmax and stage_i > smax):
            lmax, smax = L, stage_i
    return lmin, lmax, smin, smax


def get_min_load_stages(dht_map, lmin):
        min_load_stages = set()
        for stage, info in dht_map.items():
            if not info:
                continue
            if sum(p['load'] for p in info.values()) == lmin:
                min_load_stages.add(int(stage))
        return min_load_stages
