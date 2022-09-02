
#include <stdio.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <dirent.h>
#include <time.h>
#include "omp.h"
#include <assert.h>
#include <string.h>

#define REAL 0
#define IMAG 1
#define PI 3.1416
#define NUM_THREADS 4
#define RADC_DATA_SIZE_BYTES 786432
#define RADC_DATA_SIZE_ONE_ANTENNA_BYTES 262144 

const int NUM_ANTENNAS = 3;
const int sample_num = 256;
const int chirp_num = 256;
const int antenna_num = 3;
const int range_num = sample_num * 1;
const int speed_num = chirp_num * 1;

/* Hann window: hann[i] = 0.5 * (1 - cos((2* PI * i) / 256)) */
float hann[256] = {0.000000, 0.000151, 0.000602, 0.001355, 0.002408, 0.003760, 0.005412, 0.007361, 0.009607, 0.012149, 0.014984, 0.018112, 0.021530, 0.025236, 0.029228, 0.033504, 0.038060, 0.042895, 0.048006, 0.053388, 0.059040, 0.064957, 0.071136, 0.077574, 0.084266, 0.091208, 0.098397, 0.105827, 0.113495, 0.121396, 0.129525, 0.137877, 0.146447, 0.155230, 0.164221, 0.173414, 0.182804, 0.192385, 0.202151, 0.212097, 0.222216, 0.232502, 0.242950, 0.253552, 0.264303, 0.275195, 0.286224, 0.297381, 0.308660, 0.320054, 0.331556, 0.343161, 0.354859, 0.366645, 0.378511, 0.390451, 0.402456, 0.414521, 0.426636, 0.438796, 0.450993, 0.463219, 0.475468, 0.487731, 0.500002, 0.512272, 0.524536, 0.536784, 0.549011, 0.561207, 0.573367, 0.585483, 0.597547, 0.609553, 0.621492, 0.633358, 0.645144, 0.656843, 0.668447, 0.679950, 0.691344, 0.702623, 0.713780, 0.724808, 0.735700, 0.746451, 0.757053, 0.767501, 0.777787, 0.787906, 0.797852, 0.807618, 0.817199, 0.826588, 0.835781, 0.844772, 0.853555, 0.862125, 0.870477, 0.878606, 0.886507, 0.894175, 0.901605, 0.908794, 0.915736, 0.922428, 0.928866, 0.935045, 0.940962, 0.946614, 0.951996, 0.957106, 0.961941, 0.966498, 0.970773, 0.974765, 0.978471, 0.981889, 0.985016, 0.987852, 0.990393, 0.992639, 0.994589, 0.996240, 0.997593, 0.998645, 0.999398, 0.999849, 1.000000, 0.999849, 0.999398, 0.998645, 0.997592, 0.996239, 0.994588, 0.992638, 0.990392, 0.987850, 0.985015, 0.981887, 0.978469, 0.974763, 0.970771, 0.966495, 0.961938, 0.957103, 0.951993, 0.946610, 0.940959, 0.935041, 0.928862, 0.922424, 0.915732, 0.908790, 0.901601, 0.894170, 0.886502, 0.878601, 0.870472, 0.862120, 0.853550, 0.844767, 0.835776, 0.826583, 0.817193, 0.807612, 0.797846, 0.787900, 0.777781, 0.767495, 0.757047, 0.746445, 0.735694, 0.724801, 0.713773, 0.702616, 0.691337, 0.679943, 0.668440, 0.656836, 0.645137, 0.633351, 0.621485, 0.609545, 0.597540, 0.585476, 0.573360, 0.561200, 0.549003, 0.536777, 0.524528, 0.512265, 0.499994, 0.487724, 0.475461, 0.463212, 0.450986, 0.438789, 0.426629, 0.414513, 0.402449, 0.390444, 0.378504, 0.366638, 0.354852, 0.343154, 0.331549, 0.320047, 0.308653, 0.297374, 0.286217, 0.275189, 0.264296, 0.253546, 0.242943, 0.232496, 0.222210, 0.212091, 0.202145, 0.192379, 0.182798, 0.173409, 0.164216, 0.155225, 0.146442, 0.137872, 0.129520, 0.121391, 0.113491, 0.105823, 0.098392, 0.091204, 0.084261, 0.077570, 0.071132, 0.064953, 0.059036, 0.053385, 0.048002, 0.042892, 0.038058, 0.033501, 0.029226, 0.025234, 0.021528, 0.018110, 0.014983, 0.012147, 0.009606, 0.007360, 0.005411, 0.003759, 0.002407, 0.001354, 0.000602, 0.000150};

/**
 * @brief Array to hold the raw data of the packet of 1 antenna; 
 * Data type is unsigned 16-bit integer
 */
u_int16_t rawdata_antenna[3][RADC_DATA_SIZE_ONE_ANTENNA_BYTES / 2];

/**
 * @enum ASCII Codes for characters in the headers of the payload
 * We have 'R', 'A', 'D', 'C' thus far.
 */
enum relevantASCIICodes {R = 82, A = 65, D = 68, C = 67}; 

float RD_complex_mean_REAL[256][256] = {0.0};
float RD_complex_mean_IMAG[256][256] = {0.0};
float RD_complex_mean_dB[256][256] = {0.0};


int main(int argc, char** argv) {    
    FILE *fp;

    if (argc > 1) {
        printf("More than 1 file provided!\n");
        return 0;
    } 

    int n = 256; //256-point FFT
    int howmany = 256;

    /**
     * @brief Data binary file to be opened
     * The total payload length is 786432 bytes
     * The first 4 bytes are 4 ASCII characters (1 character is 1 byte) which is the header of the payload
     * The next 4 bytes are combined to form one UINT32 which is the payload length
     * The rest is data. For RADC packets, the data is sent as UINT16.
     * For RADC packets, the data consists of 3 RX chunks of 262144 bytes each.
     * The total packet size is therefore 3*262144 + 4 + 4 = 786,440 bytes
     */
    fp = fopen(argv[1],"r");

    /**
     * @brief Initializations for the FFTW library
     * Documentation can be found online: https://www.fftw.org/#documentation
     */
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(n*howmany*sizeof(fftwf_complex));
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(n*howmany*sizeof(fftwf_complex));
    fftwf_plan p;
    /**
     * @brief fftwf initialization
     * 
     */
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    /**
     * @brief Array to hold the headers of the packet
     * ASCII characters are 1 byte each
     */
    u_int8_t header[4];

    /** 
     * @param ptr − This is the pointer to a block of memory with a minimum size of size*nmemb bytes.
     * @param size − This is the size in bytes of each element to be read.
     * @param nmemb − This is the number of elements, each one with a size of size bytes.
     * @param stream − This is the pointer to a FILE object that specifies an input stream.
     * Additional note that the pointer for the file automatically advances with each call.
     */
    fread(header, 1, 4, fp);

    /**
     * @brief Confirm whether the headers are equivalent to ASCII characters 'RADC'
     */
    assert(header[0] == R && header[1] == A && header[2] == D && header[3] == C); 
    
    /**
     * @param payloadLength Length of the payload. It is a 32 bit unsigned integer (or 4 bytes).
     */
    u_int32_t payloadLength; 

    /**
     * @brief Read the payload length
     */
    fread(&payloadLength, 4, 1, fp);

    /**
     * @brief Confirm whether total payload length is correct
     */
    assert(payloadLength == RADC_DATA_SIZE_BYTES); 

    for (int antennas = 0; antennas < NUM_ANTENNAS; antennas++) {
        fread(rawdata_antenna[antennas], 2, RADC_DATA_SIZE_ONE_ANTENNA_BYTES / 2, fp);
        
        float mean_real;
        float mean_imag;
        float sum_real = 0.0;
        float sum_imag = 0.0;

        /**
         * @brief Calculaing the mean in this loop to remove DC.
         * Using OpenMP to enable parallel computing for this for loop.
         * @param omp refers to OpenMP
         * @param parallel refers to making the next section of code run in parallel
         * @param for refers to making the next for loop parallel
         * @param reduction refers to joining private copies of the variables listed after compute
         * @param private refers to having private thread copies of the variables listed
         */
        
        /**
         * @brief Calculate mean readings for real and imaginary parts
         */


        #pragma omp parallel for reduction(+:sum_real, sum_imag) num_threads(omp_get_max_threads())
        for (int i = 0; i < 65536; i++) {
            sum_real += rawdata_antenna[antennas][i * 2];
            sum_imag += rawdata_antenna[antennas][i * 2 + 1];
        }

        mean_real = sum_real / 65536;
        mean_imag = sum_imag / 65536;
        
        /**
         * @brief Minus mean from readings and multiply with Hanning window
         */
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (int i = 0; i < 65536; i++) {
            in[i][REAL] = (float) (rawdata_antenna[antennas][i * 2] - mean_real) * hann[i % 256];
            in[i][IMAG] = (float) (rawdata_antenna[antennas][i * 2 + 1] - mean_imag) * hann[i % 256];
        }
        
        /**
        * @brief File name to load and save fftwf plan. On the first run of this program,
        * the import code will need to be commented out to generate a plan to load for 
        * subsequent runs. 
        */
        char fftwfExportFilename[] = "fftwf_plan";
        
        /**
         * @brief Load plan from memory if it exists, will return 1 on sucess.
         */
        int pp = fftwf_import_wisdom_from_filename(fftwfExportFilename);
        assert(pp == 1);
        p = fftwf_plan_many_dft(1, &n, howmany, in, NULL, 1, howmany, 
            out, NULL, 1, howmany, FFTW_FORWARD, FFTW_WISDOM_ONLY);

        fftwf_execute(p);

        float mean_real_zero_dopp[256] = {0.0};
        float mean_imag_zero_dopp[256] = {0.0};

        /* Remove zero-Doppler */
        #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:mean_real_zero_dopp, mean_imag_zero_dopp) 
        for (int i = 0; i < 65536; i++) {
            mean_real_zero_dopp[i % 256] += out[i][REAL];
            mean_imag_zero_dopp[i % 256] += out[i][IMAG];
        }

        for (int i = 0; i < 256; i++) {
            mean_real_zero_dopp[i] -= mean_real_zero_dopp[i] / 256.0;
            mean_imag_zero_dopp[i] -= mean_imag_zero_dopp[i] / 256.0;
        }

        #pragma omp barrier
        /**
         * @brief So the first FFT was for columns, which is rawdata[0], rawdata[1]...
         * The second FFT is for rows, which is rawdata[0], rawdata[256], rawdata[512]... rawdata[1], 
         * rawdata[257], rawdata[513] ...
         * Therefore, I did a matrix transposition such that the new order is the transposed matrix.
         * However, there is probably a more elegant way to do this by changing the parameters of the fftw plan.
         */
        #pragma omp parallel for num_threads(omp_get_max_threads()) 
        for (int i = 0; i < 65536; i++) {
            in[i][REAL] = (out[i/256 + (i%256)*256][REAL] - mean_real_zero_dopp[i / 256]) * hann[i % 256];
            in[i][IMAG] = (out[i/256 + (i%256)*256][IMAG] - mean_imag_zero_dopp[i / 256]) * hann[i % 256];
        }
        
        p = fftwf_plan_many_dft(1, &n, howmany, in, NULL, 1, howmany, 
            out, NULL, 1, howmany, FFTW_FORWARD, FFTW_WISDOM_ONLY);
        
        fftwf_execute(p);

        /**
         * @brief Save plan for FFTW here. Will return 1 on success.
         */
        // int exportResult = fftwf_export_wisdom_to_filename(fftwfExportFilename);
        // assert(exportResult == 1);
    
    }

    
    /**
     * @brief Cleanup done here.
     */
    fclose(fp);
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
    fftwf_cleanup_threads();
    fclose(fp_txt);

    return 0;
}
