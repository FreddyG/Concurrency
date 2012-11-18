/*
  A program to generate an image of the Mandelbrot set. 

  Usage: ./mandelbrot > output
         where "output" will be a binary image, 1 byte per pixel
         The program will print instructions on stderr as to how to
         process the output to produce a JPG file.

  Michael Ashley / UNSW / 13-Mar-2003
*/

// Define the range in x and y here:

const double yMin = -1.0;
const double yMax = +1.0;
const double xMin = -2.0;
const double xMax = +0.5;

// And here is the resolution:

const double dxy = 0.005;

#include <stdio.h>
#include <limits.h>

#include <unistd.h>
#include <omp.h>
#include "timer.h"

int main(void) {
    timer_start();

    double cx, cy;
    double zx, zy, new_zx;
    unsigned char n;
    int nx, ny;

    // The Mandelbrot calculation is to iterate the equation
    // z = z*z + c, where z and c are complex numbers, z is initially
    // zero, and c is the coordinate of the point being tested. If
    // the magnitude of z remains less than 2 for ever, then the point
    // c is in the Mandelbrot set. We write out the number of iterations
    // before the magnitude of z exceeds 2, or UCHAR_MAX, whichever is
    // smaller.

    // needed because OpenMP can only parallelize for loops with integer loop
    // variables
    int cx_iter, cy_iter;
    int xmin_iter = (int)(xMin * 1000),
        ymin_iter = (int)(yMin * 1000);
    int xmax_iter = (int)(xMax * 1000),
        ymax_iter = (int)(yMax * 1000);
    int dxy_iter  = (int)(dxy * 1000);

    for (cy_iter = ymin_iter; cy_iter < ymax_iter; cy_iter += dxy_iter) {
        #pragma omp parallel for shared(cy_iter)
        for (cx_iter = xmin_iter; cx_iter <= xmax_iter; cx_iter += dxy_iter) {
            cx = ((double)cx_iter) / 1000;
            cy = ((double)cy_iter) / 1000;
            zx = 0.0; 
            zy = 0.0; 
            n = 0;
            while ((zx*zx + zy*zy < 4.0) && (n != UCHAR_MAX)) {
                new_zx = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = new_zx;
                n++;
            }

            #pragma omp critical
            {
            write (1, &n, sizeof(n)); // Write the result to stdout
            }
        }
    }

    // Now calculate the image dimensions. We use exactly the same
    // for loops as above, to guard against any potential rounding errors.

    nx = 0;
    ny = 0;
    for (cx = xMin; cx < xMax; cx += dxy) {
        nx++;
    }
    for (cy = yMin; cy < yMax; cy += dxy) {
        ny++;
    }

    fprintf (stderr, "To process the image: convert -depth 8 -size %dx%d gray:output out.jpg\n",
             nx, ny);
    double elapsed = timer_end();
    fprintf(stderr, "Time elapsed: %f seconds\n", elapsed);
    return 0;
}

