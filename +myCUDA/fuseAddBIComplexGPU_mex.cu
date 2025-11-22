#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math.h> 
#include <stdio.h> 

typedef const unsigned int cuint;
typedef const uint32_T cuint32;

int checkLastError(char * msg)
{
    cudaError_t cudaStatus = cudaGetLastError();
    if ( cudaStatus != cudaSuccess ) {
        char err[512];
        sprintf(err, "setprojection failed \n %s: %s. \n", msg, cudaGetErrorString(cudaStatus));
        mexPrintf(err);
        return 1;
    }
    return 0;
}

__global__ void addToArrayBI_c(float2 const * sarray, float2 * larray, 
                               cuint32* pos_X, cuint32* posY, 
                               float const * ul, float const * bl, float const * ur, float const * br,
                               cuint Np_px, cuint Np_py, cuint nSpotsParallel, 
                               cuint Np_ox, cuint Np_oy, cuint nSlices)  {
    // Location in a 3D matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < Np_px - 1 & idy < Np_py - 1){
        int id_large, id_small; 
        float2 sarray_val_ul, sarray_val_bl, sarray_val_ur, sarray_val_br;
        for (int iSlice = 0; iSlice < nSlices; iSlice++) {
            for (int iSpot = 0; iSpot < nSpotsParallel; iSpot++) { 
                id_large = (pos_X[iSpot] - 1 + idx) + Np_ox * (posY[iSpot] - 1 + idy) + Np_ox * Np_oy * iSlice;
                id_small = idx + Np_px * idy + Np_px * Np_py * iSpot + Np_px * Np_py * nSpotsParallel * iSlice;
    
                sarray_val_ul = sarray[id_small];
                sarray_val_bl = sarray[id_small+1];
                sarray_val_ur = sarray[id_small+Np_px];
                sarray_val_br = sarray[id_small+Np_px+1];
                atomicAdd(&larray[id_large].x, (sarray_val_ul.x*ul[iSpot] + sarray_val_bl.x*bl[iSpot] + sarray_val_ur.x*ur[iSpot] + sarray_val_br.x*br[iSpot]));
                atomicAdd(&larray[id_large].y, (sarray_val_ul.y*ul[iSpot] + sarray_val_bl.y*bl[iSpot] + sarray_val_ur.y*ur[iSpot] + sarray_val_br.y*br[iSpot]));
            }
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // input
    // prhs[0]: subVolumes, gpuArray, single -> m_obj_proj
    // 4 dims: (probNX+2, probNY+2, nSpotsParallel, nSlices) 
    // 3 dims: (probNX+2, probNY+2, nSpotsParallel)
    // 2 dims: (probNX+2, probNY+2)
    // pad 0 around
    
    // prhs[1]: object, gpuArray, single
    // 3 dims: (objNX, objNY, nSlices)
    // 2 dims: (objNX, objNY)
    
    // prhs[2]: anchorULXs, (nSpotsParallel, 1), uint32 -> m_positions_x
    // prhs[3]: anchorULYs, (nSpotsParallel, 1), uint32 -> m_positions_y

    // prhs[4]: upperLeftFactor, (nSpotsParallel, 1), uint32 -> m_ul
    // prhs[5]: bottomLeftFactor, (nSpotsParallel, 1), uint32 -> m_bl
    // prhs[6]: upperRightFactor, (nSpotsParallel, 1), uint32 -> m_uR
    // prhs[7]: bottomRightFactor, (nSpotsParallel, 1), uint32 -> m_bR
    

    int i; // for loop 
    // --- check inputs ---
    for (i = 0; i < 2; i++) {
        // gpuArray
        if ( !mxIsGPUArray(prhs[i]) ) {
            printf("Input %d is not gpu array\n", i+1);
            mexErrMsgIdAndTxt("MexError:ptycho", "Inputs must be of correct type.");
        }
    }
    // It cannot be one-dimensional 
    if ( mxGetNumberOfDimensions(prhs[0]) < 2 ) {
        printf("Input 1 must have at least two dimensions.\n");
        mexErrMsgIdAndTxt("MexError:ptycho", "wrong number of dimensions");
    }
    // It cannot be more than 3-dimensional 
    if ( mxGetNumberOfDimensions(prhs[0]) > 4 ) {
        printf("Input 1 must have at most four dimensions.\n");
        mexErrMsgIdAndTxt("MexError:ptycho", "wrong number of dimensions");
    }
    // It cannot be one-dimensional 
    if ( mxGetNumberOfDimensions(prhs[1]) < 2 ) {
        printf("Input 2 must have at least two dimensions.\n");
        mexErrMsgIdAndTxt("MexError:ptycho", "wrong number of dimensions");
    }
    // It cannot be more than 3-dimensional 
    if ( mxGetNumberOfDimensions(prhs[1]) > 3 ) {
        printf("Input 2 must have at most three dimensions.\n");
        mexErrMsgIdAndTxt("MexError:ptycho", "wrong number of dimensions");
    }
    for (i = 2; i < 4; i++){
        // Input must be of type uint32.
        if ( !mxIsUint32(prhs[i]) ) {
            printf("Input %d is not integer\n", i+1);
            mexErrMsgIdAndTxt("MexError:ptycho", "Inputs must be of correct type uint32.");
        }
    }
    for (i = 4; i < 8; i++){
        // Input must be of type single.
        if ( !mxIsSingle(prhs[i]) ) {
            printf("Input %d is not single\n", i+1);
            mexErrMsgIdAndTxt("MexError:ptycho", "Inputs must be of correct type single.");
        }
    }
    
    // --- create mxGPUArray ---
    const mxGPUArray * m_projection = mxGPUCreateFromMxArray(prhs[0]);
    const float2 * p_projection = (float2 *)mxGPUGetDataReadOnly(m_projection);

    mxGPUArray * m_object = mxGPUCopyFromMxArray(prhs[1]);
    float2 * p_object = (float2 *)mxGPUGetData(m_object);

    const mxGPUArray * m_positions_x = mxGPUCreateFromMxArray(prhs[2]);
    const uint32_T * p_positions_x = (uint32_T *)mxGPUGetDataReadOnly(m_positions_x);

    const mxGPUArray * m_positions_y = mxGPUCreateFromMxArray(prhs[3]);
    const uint32_T * p_positions_y = (uint32_T *)mxGPUGetDataReadOnly(m_positions_y);

    const mxGPUArray * m_ul = mxGPUCreateFromMxArray(prhs[4]);
    const float * p_ul = (float *)mxGPUGetDataReadOnly(m_ul);

    const mxGPUArray * m_bl = mxGPUCreateFromMxArray(prhs[5]);
    const float * p_bl = (float *)mxGPUGetDataReadOnly(m_bl);

    const mxGPUArray * m_ur = mxGPUCreateFromMxArray(prhs[6]);
    const float * p_ur = (float *)mxGPUGetDataReadOnly(m_ur);

    const mxGPUArray * m_br = mxGPUCreateFromMxArray(prhs[7]);
    const float * p_br = (float *)mxGPUGetDataReadOnly(m_br);
    
    // --- Get dimension of small and large object ---
    const unsigned int Ndims_p = (unsigned int)mxGPUGetNumberOfDimensions(m_projection);
    const mwSize * Np_p = mxGPUGetDimensions(m_projection); // probNX probNY nSpotsParallel
    unsigned int nSpotsParallel;
    if (Ndims_p == 2) {
        nSpotsParallel = 1;
    } else {
        nSpotsParallel = Np_p[2];
    }
    const unsigned int Ndims_o = (unsigned int)mxGPUGetNumberOfDimensions(m_object);
    const mwSize * Np_o = mxGPUGetDimensions(m_object); // objNX objNY
    unsigned int nSlices;
    if (Ndims_o == 2) {
        nSlices = 1;
    } else {
        nSlices = Np_o[2];
    }
    if (Ndims_p == 4 && Np_p[3] != nSlices) {
        printf("nSlices");
        mexErrMsgIdAndTxt("MexError:ptycho", "nSlices is not right");
    }
    
    const unsigned int Npos_x = mxGPUGetNumberOfElements(m_positions_x); // nSpotsParallel
    const unsigned int Npos_y = mxGPUGetNumberOfElements(m_positions_y); // nSpotsParallel
    const unsigned int Npos_ul = mxGPUGetNumberOfElements(m_ul); // nSpotsParallel
    const unsigned int Npos_bl = mxGPUGetNumberOfElements(m_bl); // nSpotsParallel
    const unsigned int Npos_ur = mxGPUGetNumberOfElements(m_ur); // nSpotsParallel
    const unsigned int Npos_br = mxGPUGetNumberOfElements(m_br); // nSpotsParallel
    if (Npos_x != nSpotsParallel || Npos_y != nSpotsParallel || Npos_ul != nSpotsParallel || Npos_bl != nSpotsParallel || Npos_ur != nSpotsParallel || Npos_br != nSpotsParallel) {
        printf("nSpotsParallel %u %u %u %u %u %u %u", Npos_x, Npos_y, Npos_ul, Npos_bl, Npos_ur, Npos_br, nSpotsParallel);
        mexErrMsgIdAndTxt("MexError:ptycho", "nSpotsParallel is not right");
    }
    
    // --- grid & block ---
    // Choose a reasonably sized number of threads in each dimension for the block.
    int const threadsPerBlockEachDim =  32;
    // Compute the thread block and grid sizes based on the board dimensions.
    int const blocksPerGrid_M = (Np_p[0] - 1 + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (Np_p[1] - 1 + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = 1;
    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);
    checkLastError("after dimThread");

    // ================== call the right kernel =================== 
    addToArrayBI_c<<<dimBlock, dimThread>>>(p_projection, p_object, 
                                            p_positions_x, p_positions_y,
                                            p_ul, p_bl, p_ur, p_br, 
                                            Np_p[0], Np_p[1], nSpotsParallel, 
                                            Np_o[0], Np_o[1], nSlices);

    checkLastError("after kernel");

    cudaThreadSynchronize();

    plhs[0] = mxGPUCreateMxArrayOnGPU(m_object);

    mxGPUDestroyGPUArray(m_projection);
    mxGPUDestroyGPUArray(m_object);
    mxGPUDestroyGPUArray(m_positions_x);
    mxGPUDestroyGPUArray(m_positions_y);

    
  return;
}




