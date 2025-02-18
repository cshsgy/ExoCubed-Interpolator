import torch
import interpolator
import time

def test_exo_to_latlon():
    # Create a sample input tensor (6*n_lyr*N*N)
    n_lyr = 2  # number of layers
    N = 48      # grid size per face
    
    # Create a test tensor with known values
    # Shape: (6, n_lyr, N, N) for 6 faces of the cube
    input_tensor = torch.arange(6*n_lyr*N*N, dtype=torch.float64).reshape(6, n_lyr, N, N).cuda()
    
    # Define output grid size
    n_lat = 100
    n_lon = n_lat * 2
    
    try:
        # Run the interpolation
        start_time = time.time()
        for i in range(100):
            output = interpolator.exo_to_latlon(input_tensor, n_lat, n_lon)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Time taken for each exo_to_latlon: {(end_time - start_time)/100 * 1000} ms")
        
        # Check output shape
        expected_shape = (n_lyr, n_lat, n_lon)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Print some basic information
        print("Test passed!")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print("\nFirst few values of input tensor:")
        print(input_tensor[0, 0, :2, :2])  # Show a small section of first face
        print("\nFirst few values of output tensor:")
        print(output[0, :2, :2])  # Show a small section of first layer
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_exo_to_latlon() 