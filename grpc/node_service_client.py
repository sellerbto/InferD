import  asyncio
import logging

import grpc
import node_service_pb2_grpc
import  node_service_pb2
from node_service_pb2_grpc import NodeServiceStub


async def node_send_tensor(stub : NodeServiceStub, tensor : node_service_pb2.TensorRequest) -> None :
    resp = await stub.SendTensor(tensor)
    print(resp.is_success)



async def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = node_service_pb2_grpc.NodeServiceStub(channel)
        print("-------------- SendTensor --------------")
        await node_send_tensor(stub, node_service_pb2.TensorRequest())



if __name__ == "__main__":
    logging.basicConfig()
    asyncio.get_event_loop().run_until_complete(run())