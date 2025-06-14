from ...simple_chain_client import SimpleChainRPCClient
import torch
import uuid

def test_chain_inference():
    """
    Test the new chain-based inference architecture.
    """
    print("=== Testing Chain-Based Inference ===")

    # Define bootstrap nodes (adjust these to your actual node addresses)
    bootstrap_nodes = [
        ("localhost", 6050),  # Node for stage 0
        ("localhost", 6051),  # Node for stage 1
        ("localhost", 6052),  # Node for stage 2
        ("localhost", 6053),  # Node for stage 3
    ]

    # Create simple chain client
    client = SimpleChainRPCClient(bootstrap_nodes)

    # Create test tensors (simulating real inference data)
    batch_size, seq_len, hidden_size = 1, 10, 512

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    cache_position = torch.arange(seq_len)
    cos = torch.randn(batch_size, seq_len, hidden_size // 32)  # Simplified
    sin = torch.randn(batch_size, seq_len, hidden_size // 32)  # Simplified

    session_id = str(uuid.uuid4())

    print(f"Session ID: {session_id}")
    print(f"Input shape: {hidden_states.shape}")

    try:
        # Test 1: Chain-based inference with cache
        print("\n--- Test 1: Chain-based inference with cache ---")

        result = client.forward_through_chain(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            cos=cos,
            sin=sin,
            session_id=session_id,
            use_cache=True
        )

        print(f"Chain inference result shape: {result.shape}")
        print(f"Cache info: {client.get_cache_info(session_id)}")

        # Test 2: Demonstrate cache recovery
        print("\n--- Test 2: Demonstrating cache usage ---")

        # Simulate a second inference step using the same session
        # This would use cached intermediate results
        result2 = client.forward_through_chain(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            cos=cos,
            sin=sin,
            session_id=session_id,
            use_cache=True
        )

        print(f"Second inference result shape: {result2.shape}")

    except Exception as e:
        print(f"Chain inference failed, trying fallback: {e}")

        # Test 3: Fallback to traditional layer-by-layer
        print("\n--- Test 3: Fallback to layer-by-layer processing ---")

        try:
            result_fallback = client.fallback_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                cos=cos,
                sin=sin,
                session_id=session_id + "_fallback"
            )

            print(f"Fallback result shape: {result_fallback.shape}")
            print(f"Fallback cache info: {client.get_cache_info(session_id + '_fallback')}")

        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")

    finally:
        # Clean up cache
        print("\n--- Cleanup ---")
        print(f"Final cache info: {client.get_cache_info(session_id)}")
        client.clear_cache()
        print("Cache cleared")

def test_cache_management():
    """
    Test cache management features.
    """
    print("\n=== Testing Cache Management ===")

    bootstrap_nodes = [("localhost", 6050)]
    client = SimpleChainRPCClient(bootstrap_nodes)

    # Create multiple sessions to test cache management
    sessions = []
    for i in range(3):
        session_id = f"test_session_{i}"
        sessions.append(session_id)

        # Simulate cached data
        client.cache[session_id] = {
            0: torch.randn(1, 10, 512),
            1: torch.randn(1, 10, 512),
        }

    print("Created test cache data for sessions:", sessions)

    for session_id in sessions:
        cache_info = client.get_cache_info(session_id)
        print(f"Session {session_id}: {cache_info}")

    # Test selective cache clearing
    client.clear_cache(sessions[0])
    print(f"After clearing {sessions[0]}:")

    for session_id in sessions:
        cache_info = client.get_cache_info(session_id)
        print(f"Session {session_id}: {cache_info}")

    # Clear all cache
    client.clear_cache()
    print("After clearing all cache:")

    for session_id in sessions:
        cache_info = client.get_cache_info(session_id)
        print(f"Session {session_id}: {cache_info}")

if __name__ == "__main__":
    print("Chain-Based Inference Test")
    print("=" * 50)

    try:
        test_chain_inference()
        test_cache_management()

        print("\n" + "=" * 50)
        print("All tests completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
