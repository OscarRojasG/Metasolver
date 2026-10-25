/*
 * main_clp.cpp
 *
 *  Created on: 29 jun. 2017
 *      Author: iaraya
 */

#include <iostream>
#include <fstream>
#include "args.hxx"
#include "clpState.h"
#include "BlockSet.h"
#include "VCS_Function.h"
#include "SpaceSet.h"
#include "Greedy.h"
#include "DoubleEffort.h"
#include "GlobalVariables.h"
#include "DataPrinter.h"

using namespace std;

int main(int argc, char** argv) {
	args::ArgumentParser parser("********* BSG-CLP *********.", "BSG Solver for CLP.");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::ValueFlag<int> _inst(parser, "int", "Instance", {'i'});
	args::ValueFlag<string> _format(parser, "string", "Format: (BR, BRw, 1C)", {'f'});
	args::ValueFlag<double> _min_fr(parser, "double", "Minimum volume occupied by a block (proportion)", {"min_fr"});
	args::ValueFlag<int> _w(parser, "int", "Beam width (nodes per level)", {'w'});
	args::ValueFlag<double> _alpha(parser, "double", "Alpha parameter", {"alpha"});
	args::ValueFlag<double> _beta(parser, "double", "Beta parameter", {"beta"});
	args::ValueFlag<double> _gamma(parser, "double", "Gamma parameter", {"gamma"});
	args::ValueFlag<double> _delta(parser, "double", "Delta parameter", {"delta"});
	args::ValueFlag<double> _p(parser, "double", "p parameter", {'p'});
	args::Positional<std::string> _file(parser, "instance-set", "The name of the instance set");

	cout.precision(8);
	try
	{
		parser.ParseCLI(argc, argv);

	}
	catch (args::Help&)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError& e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}
	catch (args::ValidationError& e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	string file=_file.Get();
	int inst=(_inst)? _inst.Get():0;
	double min_fr=(_min_fr)? _min_fr.Get():0.98;
	int w=(_w)? _w.Get():4;

	double alpha=4.0, beta=1.0, gamma=0.2, delta=1.0, p=0.04;
	if(_alpha) alpha=_alpha.Get();
	if(_beta) beta=_beta.Get();
	if(_gamma) gamma=_gamma.Get();
	if(_delta) delta=_delta.Get();
	if(_p) p=_p.Get();


	string format="BR";
	if(_format) format=_format.Get();

	clpState::Format f;
	if(format=="BR")
		f=clpState::BR;
	else if(format=="BRw")
		f=clpState::BRw;
	else if(format=="1C")
		f=clpState::_1C;

	global::TRACE = false;


	cout << "***** Creando el contenedor ****" << endl;
	cout << "File("<< format <<"): " << file << endl;
	cout << "Instance:" << inst+1 << endl;
	cout << "min_fr:" << min_fr << endl;

	double r=0.0; //0.0

    clpState* s0 = new_state(file,inst, min_fr, 10000, f);

    cout << "n_blocks:"<< s0->get_n_valid_blocks() << endl;

    VCS_Function* vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

	cout << "greedy" << endl;
    SearchStrategy *gr = new Greedy (vcs);

	cout << "copying state" << endl;
	State& s_copy= *s0->clone();

	cout << "running" << endl;

	double eval = gr->run(s_copy);
	double time = gr->get_time();

    cout << "% volume utilization" << endl;
	cout << eval*100 << endl;

	cout << "total time" << endl;
	cout << time << endl;
}