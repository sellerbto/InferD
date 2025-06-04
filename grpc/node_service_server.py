from concurrent import futures

import grpc
import logging

import node_service_pb2
import node_service_pb2_grpc
from node_service_pb2 import RequestResult


class NodeServiceServicer(node_service_pb2_grpc.NodeServiceServicer):
    """Missing associated documentation comment in .proto file."""

    def SendTensor(self, request : node_service_pb2.TensorRequest, context)\
            -> node_service_pb2.RequestResult:
        """Missing associated documentation comment in .proto file."""
        return RequestResult(is_success=True, description="test")

    def SaveTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSavedTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNodeStats(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(
        NodeServiceServicer(),
        server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()