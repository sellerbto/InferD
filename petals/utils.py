import os
import io
import torch
import grpc
from concurrent import futures
from models.qwen3.proto import qwen3_pb2, qwen3_pb2_grpc
import argparse
import yaml

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
        L = sum(p['load'] for p in peers.values())
        # print(f'node stage = {node_info.stage}, ip:host = {node_info.id}, load = {L}, stage number = {stage_i}')
        if L > lmax:
            lmax, smax = L, stage_i
        if L < lmin:
            lmin, smin = L, stage_i
    return lmin, lmax, smin, smax


def get_min_load_stages(dht_map, lmin):
        min_load_stages = set()
        for stage, info in dht_map.items():
            if not info:
                continue
            if sum(p['load'] for p in info.values()) == lmin:
                min_load_stages.add(int(stage))
        return min_load_stages

def deserialize_tensor(blob: qwen3_pb2.TensorBlob):
    buf = io.BytesIO(blob.data)
    return torch.load(buf, map_location=self.server_module.device)

def serialize_tensor(tensor: torch.Tensor):
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    return buf.getvalue()

def get_start_end_layer_by_stage(stage: int):
    with open("inferd.yaml") as f:
        cfg = yaml.safe_load(f)

    stage_to_segment = {info['stage']: (info['start_layer'], info['end_layer']) for info in cfg['stages']}
    return stage_to_segment[stage]

def create_stub(host, port, options):
    channel = grpc.insecure_channel(f"{host}:{port}", options=options)
    return qwen3_pb2_grpc.Qwen3LayerStub(channel)
