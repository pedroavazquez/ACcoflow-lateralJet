/*
 * Postprocess lateralJet_params dumps.
 *
 * First milestone:
 * - read params.txt;
 * - restore the selected Basilisk dump;
 * - exit without doing any postprocessing yet.
 *
 * Usage:
 *   ./lateralJet_postprocess [params.txt]
 *   ./lateralJet_postprocess --params params.txt
 *   ./lateralJet_postprocess --params params.txt --dump dump-file
 */

#include "grid/multigrid.h"
#include "axi.h"
#include "navier-stokes/centered.h"
#include "navier-stokes/double-projection.h"
#include "two-phase.h"
#include "tension.h"
#include "curvature.h"

#include <string.h>

#define PI (3.14159265359)
#define EPS0 (8.854e-12)

#ifndef NX
#define NX 1
#endif
#ifndef LEVEL
#define LEVEL 10
#endif
#ifndef ELECTRIC
#define ELECTRIC 1
#endif
#ifndef DI
#define DI 1
#endif

typedef struct {
	double R1;
	double R2;
	double LIN;
	double LBOX;
	double XNEUMANN;
	char DUMP_FILE[256];

	double EPSR1;
	double EPSR2;
	double SIGMA1;
	double SIGMA2;
	double CL;

	double MU1;
	double MU2;
	double OH1;
	double OH2;

	double CA1;
	double CA2;

	double FREQ;
	double BOE;
	double TELEC;
	double V0;

	double GAMMA;

	double TEND;
	double DTOUT;
	double DTDUMP;
	double DTMAXMINE;
} SimParams;

SimParams params;
double EPSA1, EPSA2;
double FREQOHM;
double FORCE;
double LX;
double AR, BR, CR;
double AI, BI, CI;

static const char *params_file = "params.txt";
static int restore_status = 1;

void init_default_params(SimParams *p)
{
	p->R1 = 1.0;
	p->R2 = 1.0;
	p->LIN = 1.0;
	p->LBOX = 1.0;
	p->XNEUMANN = -1.0;
	strcpy(p->DUMP_FILE, "dump");

	p->EPSR1 = 80.0;
	p->EPSR2 = 2.1;
	p->SIGMA1 = 3.e-3;
	p->SIGMA2 = 1.e-10;
	p->CL = 0.1;

	p->MU1 = 0.001;
	p->MU2 = 0.1;
	p->OH1 = 0.062;
	p->OH2 = 6.2;

	p->CA1 = 1.e-4;
	p->CA2 = 0.05;

	p->FREQ = 100.e3;
	p->BOE = 0.0;
	p->TELEC = 0.0;
	p->V0 = 1.0;

	p->GAMMA = 1.0;

	p->TEND = 140.0;
	p->DTOUT = 10.0;
	p->DTDUMP = 5.0;
	p->DTMAXMINE = 0.01;
}

void compute_derived_params(SimParams *p)
{
	if (p->LIN <= 0.)
		p->LIN = p->R2;
	if (p->XNEUMANN <= 0.)
		p->XNEUMANN = 2.*p->LIN;
	if (p->LBOX <= 0.)
		p->LBOX = p->R2;

	EPSA1 = p->EPSR1*EPS0;
	EPSA2 = p->EPSR2*EPS0;
	FREQOHM = p->SIGMA1/EPSA1;
	FORCE = 2.0*p->BOE/p->CA2;
	LX = p->LBOX*NX;

	AR = 1.0;
	BR = p->EPSR2/p->CL;
	CR = p->V0;
	AI = 1.0;
	BI = p->EPSR2/p->CL;
	CI = 0.0;
}

int load_params(const char *filename, SimParams *p)
{
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Error: cannot open %s for reading\n", filename);
		return -1;
	}

	char line[256];
	char name[64];
	char str_value[256];
	double value;

	while (fgets(line, sizeof(line), fp)) {
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
			continue;

		if (sscanf(line, "%63s = %lf", name, &value) == 2) {
			if (strcmp(name, "R1") == 0) p->R1 = value;
			else if (strcmp(name, "R2") == 0) p->R2 = value;
			else if (strcmp(name, "LIN") == 0) p->LIN = value;
			else if (strcmp(name, "XNEUMANN") == 0) p->XNEUMANN = value;
			else if (strcmp(name, "LBOX") == 0) p->LBOX = value;
			else if (strcmp(name, "EPSR1") == 0) p->EPSR1 = value;
			else if (strcmp(name, "EPSR2") == 0) p->EPSR2 = value;
			else if (strcmp(name, "SIGMA1") == 0) p->SIGMA1 = value;
			else if (strcmp(name, "SIGMA2") == 0) p->SIGMA2 = value;
			else if (strcmp(name, "CL") == 0) p->CL = value;
			else if (strcmp(name, "MU1") == 0) p->MU1 = value;
			else if (strcmp(name, "MU2") == 0) p->MU2 = value;
			else if (strcmp(name, "OH1") == 0) p->OH1 = value;
			else if (strcmp(name, "OH2") == 0) p->OH2 = value;
			else if (strcmp(name, "CA1") == 0) p->CA1 = value;
			else if (strcmp(name, "CA2") == 0) p->CA2 = value;
			else if (strcmp(name, "FREQ") == 0) p->FREQ = value;
			else if (strcmp(name, "BOE") == 0) p->BOE = value;
			else if (strcmp(name, "TELEC") == 0) p->TELEC = value;
			else if (strcmp(name, "V0") == 0) p->V0 = value;
			else if (strcmp(name, "GAMMA") == 0) p->GAMMA = value;
			else if (strcmp(name, "TEND") == 0) p->TEND = value;
			else if (strcmp(name, "DTOUT") == 0) p->DTOUT = value;
			else if (strcmp(name, "DTDUMP") == 0) p->DTDUMP = value;
			else if (strcmp(name, "DTMAXMINE") == 0) p->DTMAXMINE = value;
		}
		else if (sscanf(line, "%63s = %255s", name, str_value) == 2) {
			if (strcmp(name, "DUMP_FILE") == 0) {
				strncpy(p->DUMP_FILE, str_value, sizeof(p->DUMP_FILE) - 1);
				p->DUMP_FILE[sizeof(p->DUMP_FILE) - 1] = '\0';
			}
		}
	}

	fclose(fp);
	fprintf(stdout, "# Parameters loaded from %s\n", filename);
	return 0;
}

void print_run_context(const SimParams *p)
{
	fprintf(stdout, "# ===== Postprocessing Context =====\n");
	fprintf(stdout, "# Params file = %s\n", params_file);
	fprintf(stdout, "# Dump file   = %s\n", p->DUMP_FILE);
	fprintf(stdout, "# Compile-time: NX=%d, LEVEL=%d, ELECTRIC=%d, DI=%d\n",
	        NX, LEVEL, ELECTRIC, DI);
	fprintf(stdout, "# R1=%g R2=%g LIN=%g LBOX=%g LX=%g\n",
	        p->R1, p->R2, p->LIN, p->LBOX, LX);
	fprintf(stdout, "# CA1=%g CA2=%g BOE=%g FREQ=%g\n",
	        p->CA1, p->CA2, p->BOE, p->FREQ);
	fprintf(stdout, "# ==================================\n");
}

#define robin(a,b,c) ((dirichlet ((c)*Delta/(2*(b) + (a)*Delta))) + ((neumann (0))*((2*(b) - (a)*Delta)/(2*(b) + (a)*Delta) + 1.)))
#define sig(f) ((clamp(f,0,1)*params.SIGMA1 + (1.-clamp(f,0,1))*params.SIGMA2))
#define epsA(f) (clamp(f,0,1)*EPSA1 + (1.-clamp(f,0,1))*EPSA2)

double Uin1(double y)
{
	return params.CA1*params.MU2/(params.CA2*params.MU1);
}

double Uin2(double x)
{
	return -1.0;
}

scalar phiR[], phiI[];
face vector sigma[], epsilonA[];
face vector epsilonRWGF[];
face vector alphaR[], alphaI[];
face vector EfR[];
face vector EfI[];
vector ER[];
vector EI[];
scalar qR[];
scalar qI[];
vector FAC[];
vector FACFORCE[];
scalar kappa[];
vector FC[];

u.n[left] = dirichlet(Uin1(y));
u.t[left] = dirichlet(0.0);
u.n[top] = (x <= params.LIN ? dirichlet(Uin2(x)) : dirichlet(0.));
u.t[top] = dirichlet(0.0);
u.n[right] = neumann(0.0);
u.t[right] = neumann(0.0);
u.n[bottom] = neumann(0.0);
u.t[bottom] = neumann(0.0);

p[right] = dirichlet(0.);
pf[right] = dirichlet(0.);

phiR[left] = dirichlet(0.);
phiI[left] = dirichlet(0.);
phiR[right] = neumann(0.);
phiI[right] = neumann(0.);
phiR[bottom] = neumann(0.);
phiI[bottom] = neumann(0.);
#if DI
phiR[top] = (x <= params.XNEUMANN ? neumann(0.) : robin(AR, BR, CR));
phiI[top] = (x <= params.XNEUMANN ? neumann(0.) : robin(AI, BI, CI));
#else
phiR[top] = (x <= params.XNEUMANN ? neumann(0.) : dirichlet(params.V0));
phiI[top] = (x <= params.XNEUMANN ? neumann(0.) : dirichlet(0.));
#endif

f[left] = (y < params.R1 ? 1.0 : 0.);
f[top] = 0.;

static void usage(const char *program)
{
	fprintf(stderr, "Usage: %s [--params params.txt] [--dump dump-file]\n", program);
}

int main(int argc, char *argv[])
{
	init_default_params(&params);

	const char *dump_file_override = NULL;
	for (int a = 1; a < argc; a++) {
		if (strcmp(argv[a], "--params") == 0 || strcmp(argv[a], "-p") == 0) {
			if (++a >= argc) {
				usage(argv[0]);
				return 1;
			}
			params_file = argv[a];
		}
		else if (strcmp(argv[a], "--dump") == 0 || strcmp(argv[a], "-d") == 0) {
			if (++a >= argc) {
				usage(argv[0]);
				return 1;
			}
			dump_file_override = argv[a];
		}
		else if (argv[a][0] == '-') {
			fprintf(stderr, "Unknown option: %s\n", argv[a]);
			usage(argv[0]);
			return 1;
		}
		else {
			params_file = argv[a];
		}
	}

	if (load_params(params_file, &params) != 0)
		return 1;

	if (dump_file_override) {
		strncpy(params.DUMP_FILE, dump_file_override, sizeof(params.DUMP_FILE) - 1);
		params.DUMP_FILE[sizeof(params.DUMP_FILE) - 1] = '\0';
	}

	compute_derived_params(&params);
	print_run_context(&params);

	dimensions(nx = NX, ny = 1, nz = 1);
	size(LX);
	init_grid(1 << LEVEL);
	origin(0., 0.);

	rho1 = 1., rho2 = 1.;
	mu1 = params.MU1/params.MU2, mu2 = 1.;
	f.sigma = params.GAMMA/params.CA2;
	stokes = true;

	TOLERANCE = 1.e-6 [*];
	DT = params.DTMAXMINE;

	run();

	return restore_status;
}

event init(t = 0)
{
	fprintf(stdout, "# Restoring from: %s\n", params.DUMP_FILE);

	if (!restore(file = params.DUMP_FILE)) {
		fprintf(stderr, "Error: could not restore dump file %s\n", params.DUMP_FILE);
		restore_status = 1;
		return 1;
	}

	foreach()
		cm[] = y;
	cm[top] = dirichlet(y);
	cm[bottom] = dirichlet(y);
	foreach_face()
		fm.x[] = max(y, 1./HUGE);
	fm.t[top] = dirichlet(y);
	fm.t[bottom] = dirichlet(y);

	restore_status = 0;
	fprintf(stdout, "# Dump restored successfully at t = %.15g, i = %d\n", t, i);


	// Save fields for visualization
	{
		FILE *fp = fopen("fields", "w");
		if (fp) {
			output_field({f, u, phiR, ER, qR, FC, FAC, FACFORCE},fp, n=2*N);
			//output_field({f, u, phiR, phiI, ER, EI, qR, qI, FC, FAC, FACFORCE},fp, 5*N,bilinear);}
		}
		else {
			fprintf(stderr, "Error: Cannot open fields for writing\n");
		}
		fclose(fp);
		fprintf(stdout, "# Saved fields for visualization: fields\n");
	}

	// save facets
	{
		char filename[256];
		sprintf(filename, "facets-%04d.dat", pid());

		FILE *ff = fopen(filename, "w");
		if (ff) {
			output_facets(f, ff);
		} else {
			fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
		}
		fclose(ff);
		fprintf(stdout, "# Saved facets for visualization: %s\n", filename);
	}

	return 1;
}
