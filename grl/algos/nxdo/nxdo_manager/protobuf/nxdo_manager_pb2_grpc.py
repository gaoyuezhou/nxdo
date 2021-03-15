# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import nxdo_manager_pb2 as nxdo__manager__pb2


class NXDOManagerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetLogDir = channel.unary_unary(
                '/NXDOManager/GetLogDir',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=nxdo__manager__pb2.NXDOString.FromString,
                )
        self.GetManagerMetaData = channel.unary_unary(
                '/NXDOManager/GetManagerMetaData',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=nxdo__manager__pb2.NXDOMetadata.FromString,
                )
        self.ClaimNewActivePolicyForPlayer = channel.unary_unary(
                '/NXDOManager/ClaimNewActivePolicyForPlayer',
                request_serializer=nxdo__manager__pb2.NXDOPlayer.SerializeToString,
                response_deserializer=nxdo__manager__pb2.NXDONewBestResponseParams.FromString,
                )
        self.SubmitFinalBRPolicy = channel.unary_unary(
                '/NXDOManager/SubmitFinalBRPolicy',
                request_serializer=nxdo__manager__pb2.NXDOPolicyMetadataRequest.SerializeToString,
                response_deserializer=nxdo__manager__pb2.NXDOConfirmation.FromString,
                )
        self.IsPolicyFixed = channel.unary_unary(
                '/NXDOManager/IsPolicyFixed',
                request_serializer=nxdo__manager__pb2.NXDOPlayerAndPolicyNum.SerializeToString,
                response_deserializer=nxdo__manager__pb2.NXDOConfirmation.FromString,
                )


class NXDOManagerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetLogDir(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetManagerMetaData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ClaimNewActivePolicyForPlayer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitFinalBRPolicy(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def IsPolicyFixed(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NXDOManagerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetLogDir': grpc.unary_unary_rpc_method_handler(
                    servicer.GetLogDir,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=nxdo__manager__pb2.NXDOString.SerializeToString,
            ),
            'GetManagerMetaData': grpc.unary_unary_rpc_method_handler(
                    servicer.GetManagerMetaData,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=nxdo__manager__pb2.NXDOMetadata.SerializeToString,
            ),
            'ClaimNewActivePolicyForPlayer': grpc.unary_unary_rpc_method_handler(
                    servicer.ClaimNewActivePolicyForPlayer,
                    request_deserializer=nxdo__manager__pb2.NXDOPlayer.FromString,
                    response_serializer=nxdo__manager__pb2.NXDONewBestResponseParams.SerializeToString,
            ),
            'SubmitFinalBRPolicy': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitFinalBRPolicy,
                    request_deserializer=nxdo__manager__pb2.NXDOPolicyMetadataRequest.FromString,
                    response_serializer=nxdo__manager__pb2.NXDOConfirmation.SerializeToString,
            ),
            'IsPolicyFixed': grpc.unary_unary_rpc_method_handler(
                    servicer.IsPolicyFixed,
                    request_deserializer=nxdo__manager__pb2.NXDOPlayerAndPolicyNum.FromString,
                    response_serializer=nxdo__manager__pb2.NXDOConfirmation.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'NXDOManager', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NXDOManager(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetLogDir(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NXDOManager/GetLogDir',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            nxdo__manager__pb2.NXDOString.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetManagerMetaData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NXDOManager/GetManagerMetaData',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            nxdo__manager__pb2.NXDOMetadata.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClaimNewActivePolicyForPlayer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NXDOManager/ClaimNewActivePolicyForPlayer',
            nxdo__manager__pb2.NXDOPlayer.SerializeToString,
            nxdo__manager__pb2.NXDONewBestResponseParams.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitFinalBRPolicy(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NXDOManager/SubmitFinalBRPolicy',
            nxdo__manager__pb2.NXDOPolicyMetadataRequest.SerializeToString,
            nxdo__manager__pb2.NXDOConfirmation.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def IsPolicyFixed(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NXDOManager/IsPolicyFixed',
            nxdo__manager__pb2.NXDOPlayerAndPolicyNum.SerializeToString,
            nxdo__manager__pb2.NXDOConfirmation.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
