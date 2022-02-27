#include <stdio.h>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

#define CHANNEL 3
#define BLOCK_DIM_X (30)
#define BLOCK_DIM_Y (30)
#define AUGMENTED_X (BLOCK_DIM_X+2)
#define AUGMENTED_Y (BLOCK_DIM_Y+2)

struct Image {
  int width;
  int height;
  unsigned int bytes;
  unsigned char *img;
  unsigned char *dev_img;
};


// Reads a color PPM image file (name provided), and
// saves data in the provided Image structure.
// The max_col_val is set to the value read from the
// input file. This is used later for writing output image.
int readInpImg (const char * fname, Image & source, int & max_col_val){

  FILE *src;

  if (!(src = fopen(fname, "rb")))
  {
      printf("Couldn't open file %s for reading.\n", fname);
      return 1;
  }

  char p,s;
  fscanf(src, "%c%c\n", &p, &s);
  if (p != 'P' || s != '6')   // Is it a valid format?
  {
      printf("Not a valid PPM file (%c %c)\n", p, s);
      exit(1);
  }

  fscanf(src, "%d %d\n", &source.width, &source.height);
  fscanf(src, "%d\n", &max_col_val);

  int pixels = source.width * source.height;
  source.bytes = pixels * 3;  // 3 => colored image with r, g, and b channels
  source.img = (unsigned char *)malloc(source.bytes);
  if (fread(source.img, sizeof(unsigned char), source.bytes, src) != source.bytes)
    {
       printf("Error reading file.\n");
       exit(1);
    }
  fclose(src);
  return 0;
}

// Write a color image into a file (name provided) using PPM file format.
// Image structure represents the image in the memory.
int writeOutImg(const char * fname, const Image & roted, const int max_col_val){

  FILE *out;
  if (!(out = fopen(fname, "wb")))
  {
      printf("Couldn't open file for output.\n");
      return 1;
  }
  fprintf(out, "P6\n%d %d\n%d\n", roted.width, roted.height, max_col_val);
  if (fwrite(roted.img, sizeof(unsigned char), roted.bytes , out) != roted.bytes)
  {
      printf("Error writing file.\n");
      return 1;
  }
  fclose(out);
  return 0;
}

__global__ void blur(unsigned char * in,unsigned char* out,int w,int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h){
        int pixValR = 0;
        int pixValG = 0;
        int pixValB = 0;
        int count = 0;
        for(int dr = -1;dr <=1;dr++){
            for(int dc = -1;dc <=1;dc++){
                int curRow = row + dr;
                int curCol = col + dc;
                if (curRow >=0 && curRow < h && curCol >=0 && curCol < w){
                    int rgbOffset = (curRow*w + curCol)*3;
                    pixValR += in[rgbOffset];
                    pixValG += in[rgbOffset+1];
                    pixValB += in[rgbOffset+2];
                    count++;
                }
            }
        }
        int offset = (row*w+col)*3;
        out[offset] = (unsigned char)(pixValR/count);
        out[offset+1] = (unsigned char)(pixValG/count);
        out[offset+2] = (unsigned char)(pixValB/count);
    }
}
__global__ void blurShared(unsigned char* in, unsigned char* out, int w, int h) {
    int filterRow, filterCol;
    int cornerRow, cornerCol;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bdx = blockDim.x; int bdy = blockDim.y;
    int row = by * (bdy - 2) + ty;
    int col = bx * (bdx - 2) + tx;
    if ((row < h + 1) && (col < w + 1)) {

        __shared__ unsigned char tile[AUGMENTED_Y][AUGMENTED_X][3];

        // load into shared memory
        int imgRow = row - 1;
        int imgCol = col - 1;
        if ((imgRow < h) && (imgCol < w) && (imgRow >= 0) && (imgCol >= 0)) {
            int rgbOffset = (imgRow*w + imgCol)*3;
            tile[ty][tx][0] = in[rgbOffset];
            tile[ty][tx][1] = in[rgbOffset+1];
            tile[ty][tx][2] = in[rgbOffset+2];
        }
        else {
            tile[ty][tx][0] = 0;
            tile[ty][tx][1] = 0;
            tile[ty][tx][2] = 0;
        }

        __syncthreads();

        int pixValR = 0;
        int pixValG = 0;
        int pixValB = 0;
        int count = 0;

        
        if ((tx >= 1) && (ty >= 1) && (ty < bdy - 1) && (tx < bdx - 1)) {

            cornerRow = ty - 1;
            cornerCol = tx - 1;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    filterRow = cornerRow + i;
                    filterCol = cornerCol + j;

                    if ((filterRow >= 0) && (filterRow <= h) && (filterCol >= 0) && (filterCol <= w)) {
                        pixValR += tile[filterRow][filterCol][0];
                        pixValG += tile[filterRow][filterCol][1];
                        pixValB += tile[filterRow][filterCol][2];
                        count++;
                    }
                }
            }
            int offset = (imgRow*w + imgCol)*3;
            out[offset] = (unsigned char)(pixValR/count);
            out[offset+1] = (unsigned char)(pixValG/count);
            out[offset+2] = (unsigned char)(pixValB/count);
        }
    }
}
int main(int argc, char **argv)
{

  if (argc != 2)
  {
      printf("Usage: exec filename\n");
      exit(1);
  }
  char *fname = argv[1];
  Image source;
  Image source1;
  int max_col_val;
  if (readInpImg(fname, source, max_col_val) != 0)  exit(1);
  if (readInpImg(fname, source1, max_col_val) != 0)  exit(1);
  unsigned char *d_img;
  unsigned char *d_img_res;
  unsigned char *d_img_res1;
  int size = source.bytes * sizeof(char);
  cudaMalloc((void **)&d_img,size);
  cudaMalloc((void **)&d_img_res,size);
  cudaMalloc((void **)&d_img_res1,size);
  cudaMemcpy(d_img,source.img,size,cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(1.0*source.width/BLOCK_DIM_X),ceil(1.0*source.height/BLOCK_DIM_Y),1);
  dim3 dimBlock(BLOCK_DIM_X,BLOCK_DIM_Y,1);
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  blur<<<dimGrid,dimBlock>>>(d_img,d_img_res,source.width,source.height);
  cudaMemcpy(source.img,d_img_res,size,cudaMemcpyDeviceToHost);
  cudaEventRecord(stop,0);
  float costtime;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&costtime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("image size (%d,%d) with block dim:(%d,%d) common using time: %f\n",source.width,source.height,BLOCK_DIM_X,BLOCK_DIM_Y,costtime);
  dim3 dimGridShared(ceil(1.0*source.width/BLOCK_DIM_X),ceil(1.0*source.height/BLOCK_DIM_Y),1);
  dim3 dimBlockShared(AUGMENTED_X,AUGMENTED_Y,1);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  blurShared<<<dimGridShared,dimBlockShared>>>(d_img,d_img_res1,source.width,source.height);
  //for(int i=0;i<5;i++){//test the change of RGB value
  //for(int j=0;j<5;j++){
  //printf("%d ",source.img[(i*source.width+j)*3]);
  //}
  //printf("\n");
  //}
  //printf("\n");
  cudaMemcpy(source1.img,d_img_res1,size,cudaMemcpyDeviceToHost);
  cudaEventRecord(stop,0);
  float costtime1;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&costtime1,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("image size (%d,%d) with block dim:(%d,%d) shared using time: %f\n",source.width,source.height,BLOCK_DIM_X,BLOCK_DIM_Y,costtime1);
  //for(int i=0;i<5;i++){//test the change of RGB value
  //for(int j=0;j<5;j++){
  //printf("%d ",source.img[(i*source.width+j)*3]);
  //}
  //printf("\n");
  //}
  //printf("\n");
  for(int i=1;i<source.height-1;i++){
    for(int j=1;j<source.width-1;j++){
      if(source.img[(i*source.width+j)*3]!=source1.img[(i*source.width+j)*3]){
          printf("%d,%d,%d\n",i,source.img[i],source1.img[i]);
          break;
      }
    }
  }

  // Write the output file
  if (writeOutImg("blur.ppm", source, max_col_val) != 0) // For demonstration, the input file is written to a new file named "roted.ppm"
   exit(1);

  free(source.img);

  exit(0);
}
