/******************************************************************************
 *
 * File:           preader.c
 *
 * Created:        29/05/2006
 *
 * Author:         Pavel Sakov
 *                 CSIRO Marine Research
 *
 * Purpose:        Serial "reader" of the output point locations
 *
 * Description:    The `reader' object enables the client code to "read" one
 *                 output point at a time, either from a file or from a virtual
 *                 nx x ny grid. As a result, using `reader' makes it possible
 *                 to interpolate on a point-by-point basis, avoiding allocation
 *                 of the whole output array in memory.
 *
 * Revisions:      None
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "spd/nn/config.h"
#include "spd/nn/nan.h"
#include "spd/nn/delaunay.h"
#include "spd/nn/nn.h"
#include "spd/nn/nn_internal.h"
#include "spd/nn/preader.h"

namespace spdlib{ namespace nn{

    #define BUFSIZE 1024

    typedef struct {
        char* fname;
        FILE* f;
        int n;
        point p;
    } reader;

    typedef struct {
        int nx;
        int ny;
        int nmax;
        double stepx;
        double stepy;
        double x0;
        int n;
        point p;
    } grid;

    struct preader {
        grid* g;
        reader* r;
    };

    static grid* grid_create(double xmin, double xmax, double ymin, double ymax, int nx, int ny)
    {
        grid* g = NULL;

        if (nx < 1 || ny < 1)
            return NULL;

        g = (grid*) malloc(sizeof(grid));
        g->nx = nx;
        g->ny = ny;
        g->nmax = nx * ny;

        g->stepx = (nx > 1) ? (xmax - xmin) / (nx - 1) : 0.0;
        g->stepy = (ny > 1) ? (ymax - ymin) / (ny - 1) : 0.0;
        g->x0 = (nx > 1) ? xmin : (xmin + xmax) / 2.0;
        g->p.y = (ny > 1) ? ymin - g->stepy : (ymin + ymax) / 2.0;
        g->n = 0;

        return g;
    }

    static point* grid_getpoint(grid* g)
    {
        if (g->n >= g->nmax)
            return NULL;

        if (g->n % g->nx == 0) {
            g->p.x = g->x0;
            g->p.y += g->stepy;
        } else
            g->p.x += g->stepx;

        g->n++;

        return &g->p;
    }

    static void grid_destroy(grid* g)
    {
        free(g);
    }

    static reader* reader_create(char* fname)
    {
        reader* r = (reader*) malloc(sizeof(reader));

        r->fname = NULL;
        r->f = NULL;
        r->p.x = NaN;
        r->p.y = NaN;
        r->p.z = NaN;
        r->n = 0;

        if (fname == NULL) {
            r->fname = strdup("stdin");
            r->f = stdin;
        } else {
            if (strcmp(fname, "stdin") == 0 || strcmp(fname, "-") == 0) {
                r->fname = strdup("stdin");
                r->f = stdin;
            } else {
                r->fname = strdup(fname);
                r->f = fopen(fname, "r");
                if (r->f == NULL)
                    nn_quit("%s: %s\n", fname, strerror(errno));
            }
        }

        return r;
    }

    static point* reader_getpoint(reader* r)
    {
        char buf[BUFSIZE];
        char seps[] = " ,;\t";
        char* token;
        point* p = &r->p;

        if (r->f == NULL)
            return NULL;

        while (1) {
            if (fgets(buf, BUFSIZE, r->f) == NULL) {
                if (r->f != stdin)
                    if (fclose(r->f) != 0)
                        nn_quit("%s: %s\n", r->fname, strerror(errno));
                r->f = NULL;

                return NULL;
            }

            if (buf[0] == '#')
                continue;
            if ((token = strtok(buf, seps)) == NULL)
                continue;
            if (!str2double(token, &p->x))
                continue;
            if ((token = strtok(NULL, seps)) == NULL)
                continue;
            if (!str2double(token, &p->y))
                continue;
            r->n++;

            return p;
        }
    }

    static void reader_destroy(reader* r)
    {
        if (r->f != stdin && r->f != NULL)
            if (fclose(r->f) != 0)
                nn_quit("%s: %s\n", r->fname, strerror(errno));
        free(r->fname);
        free(r);
    }

    preader* preader_create1(double xmin, double xmax, double ymin, double ymax, int nx, int ny)
    {
        preader* pr = (preader*) malloc(sizeof(preader));

        pr->r = NULL;
        pr->g = grid_create(xmin, xmax, ymin, ymax, nx, ny);

        return pr;
    }

    preader* preader_create2(char* fname)
    {
        preader* pr = (preader*) malloc(sizeof(preader));

        pr->g = NULL;
        pr->r = reader_create(fname);

        return pr;
    }

    point* preader_getpoint(preader* pr)
    {
        if (pr->g != NULL)
            return grid_getpoint(pr->g);
        else
            return reader_getpoint(pr->r);
    }

    void preader_destroy(preader* pr)
    {
        if (pr->g != NULL)
            grid_destroy(pr->g);
        else
            reader_destroy(pr->r);

        free(pr);
    }
}}
