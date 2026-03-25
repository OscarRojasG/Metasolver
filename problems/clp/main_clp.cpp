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
#include "clpStatekd.h"
#include "BlockSet.h"
#include "BSG_midBSG.h"
#include "VCS_Function.h"
#include "SpaceSet.h"
#include "Greedy.h"
#include "DoubleEffort.h"
#include "GlobalVariables.h"
#include "BSG.h"
#include "BlockMetrics.h"
#include "DataPrinter.h"

using namespace std;

// para ejecutar (menos de 30 tipos de caja): BSG_CLP problems/clp/benchs/BR/BR7.txt 1 1.0 30 4.0 1.0 0.2 0.04 1.0 0.0 0.0 0 0
// para ejecutar (mas de 30 tipos de caja): BSG_CLP problems/clp/benchs/BR/BR8.txt 1 0.98 30 4.0 1.0 0.2 0.04 1.0 0.0 0.0 0 0




void dfsPrintChild(const State* node, ofstream& file){
	file << "{ "<<endl;
	file<<"\t \"name\":\""<<node->get_id()<<"\",";
	file<<"\t \"parent\":\""<<node->get_parent()->get_id() <<"\",";
	file<<"\t \"value\":\""<<node->get_value() <<"\",";
	file<<"\t \"sd\":\"\",";
	file<<"\t \"mcts_value\":\"\",";
	file<<"\t \"stimated_sd\":\"\",";
	file<<"\t \"ponderated_sd\":\"\",";
	file<<"\t \"depth\":\"\",";
	file<<"\t \"num_visit\":\"\"";
	file<< "\t,\"simulations\":[]"<<endl;


	if(!node->get_children().empty()){
		file<< "\t,\"children\":["<<endl;
		for(auto c:node->get_children()){
			dfsPrintChild(c,file);
			if(c!=node->get_children().back())
			    file<<","<<endl;
		}
		file<<"]";
	}
	file<<"}"<<endl;
}

void pointsToTxt(State* root, int it) {
	ofstream myfile("problems/clp/tree_plot/flare"+std::to_string(it)+".json");
	dfsPrintChild(root,myfile);
}

int main(int argc, char** argv){


	args::ArgumentParser parser("********* BSG-CLP *********.", "BSG Solver for CLP.");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::ValueFlag<int> _inst(parser, "int", "Instance", {'i'});
	args::ValueFlag<string> _format(parser, "string", "Format: (BR, BRw, 1C)", {'f'});
	args::ValueFlag<double> _min_fr(parser, "double", "Minimum volume occupied by a block (proportion)", {"min_fr"});
	args::ValueFlag<int> _maxtime(parser, "int", "Timelimit", {'t', "timelimit"});
	args::ValueFlag<int> _w(parser, "int", "Beam width (nodes per level)", {'w'});
	args::ValueFlag<int> _seed(parser, "int", "Random seed", {"seed"});
	args::ValueFlag<double> _alpha(parser, "double", "Alpha parameter", {"alpha"});
	args::ValueFlag<double> _beta(parser, "double", "Beta parameter", {"beta"});
	args::ValueFlag<double> _gamma(parser, "double", "Gamma parameter", {"gamma"});
	args::ValueFlag<double> _delta(parser, "double", "Delta parameter", {"delta"});
	args::ValueFlag<double> _p(parser, "double", "p parameter", {'p'});
	args::Flag _json(parser, "double", "json output tuple", {"json"});
	args::Flag _verbose(parser, "layout", "Show the actions to reach the solution", {"verbose"});
	args::ValueFlag<int> _verbose2(parser, "layout", "Verbose v2", {"verbose2"});
	args::Flag _plot(parser, "double", "plot tree", {"plot"});
	args::Flag fsb(parser, "fsb", "full-support blocks", {"fsb"});
	args::Flag trace(parser, "trace", "Trace", {"trace"});
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
	int maxtime=(_maxtime)? _maxtime.Get():100;

	double alpha=4.0, beta=1.0, gamma=0.2, delta=1.0, p=0.04;
	if(_maxtime) maxtime=_maxtime.Get();
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

	int seed=(_seed)? _seed.Get():1;
	srand(seed);

	global::TRACE = trace;

// cout << "cargando la instancia..." << endl;

//a las cajas se les inicializan sus pesos en 1

	cout << "***** Creando el contenedor ****" << endl;
	cout << "File("<< format <<"): " << file << endl;
	cout << "Instance:" << inst+1 << endl;
	cout << "min_fr:" << min_fr << endl;

	if (_maxtime)
		cout << "Maxtime:" << maxtime << endl;
	else
		cout << "Beam width:" << w << endl;

	double r=0.0; //0.0

    Block::FSB=fsb;
    clpState* s0 = new_state(file,inst, min_fr, 10000, f);

    cout << "n_blocks:"<< s0->get_n_valid_blocks() << endl;

    VCS_Function* vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont,
    alpha, beta, gamma, p, delta, 0.0, r);

	cout << "greedy" << endl;
    SearchStrategy *gr = new Greedy (vcs);

	cout << "bsg" << endl;
    BSG *bsg= new BSG(vcs,*gr, w, 0.0, 0, _plot);

	cout << "copying state" << endl;
	State& s_copy= *s0->clone();

	cout << "running" << endl;

	SearchStrategy *ss;
	double eval;
	// SOLO SI PARÁMETRO T ESTÁ ESTABLECIDO
	if(_maxtime) {
		clock_t begin_time = clock();
		ss = new DoubleEffort(*bsg);
		eval = ss->run(s_copy, maxtime, begin_time);
	} else {
		ss = bsg;
		eval = ss->run(s_copy);
	}
	double time = ss->get_time();

    cout << "% volume utilization" << endl;
	cout << eval*100 << endl;

	cout << "total time" << endl;
	cout << time << endl;

	ss->get_time();

	clpState* s00 = dynamic_cast<clpState*> (s0->clone());
	DataPrinter printer(s00, vcs, w);

  	if (_verbose || _verbose2) {
		cout << "BLOCKS" << endl;
		printer.print_blocks();
		cout << endl;

		list<const Action*>& actions= dynamic_cast<const clpState*>(ss->get_best_state())->get_path();

		cout << "SOLVE STEPS" << endl;

		for(auto action:actions) {
			cout << "Actions" << endl;
			printer.print_actions();

			cout << "Placed" << endl;
			printer.print_placed();

			const clpAction* clp_action = dynamic_cast<const clpAction*> (action);

			cout << "Space" << endl;
			printer.print_space();	

			cout << "Selected Block" << endl;
			cout << clp_action->block.id << endl;

			cout << "Volume" << endl;
			printer.print_volume();

			cout << endl;

			s00->transition(*action);
		}
	}
}
