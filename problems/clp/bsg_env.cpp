#include <iostream>
#include <fstream>
#include "args.hxx"
#include "clpState.h"
#include "VCS_Function.h"
#include "BSG.h"
#include "Greedy.h"
#include "BlockMetrics.h"
#include "PathBuilder.h"
#include "DataPrinter.h"

using namespace std;
using namespace metasolver;

// define global TRACE flag used by some modules
bool metasolver::global::TRACE = false;


int main(int argc, char** argv){
	args::ArgumentParser parser("********* BSG-ENV *********.", "BSG Environment for CLP.");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::ValueFlag<int> _inst(parser, "int", "Instance", {'i'});
	args::ValueFlag<int> _w(parser, "int", "Beam width (nodes per level)", {'w'});
	args::Flag fsb(parser, "fsb", "full-support blocks", {"fsb"});
	args::Positional<std::string> _file(parser, "instance-set", "The name of the instance set");

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
	double min_fr=0.98;
	int w=(_w)? _w.Get():4;

	double alpha=4.0, beta=1.0, gamma=0.2, delta=1.0, p=0.04;
	int seed=1;

	double r=0.0;

    Block::FSB=fsb;
	clpState::Format f = clpState::BR;

    clpState* s0 = new_state(file, inst, min_fr, 10000, f);

    VCS_Function* vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

	PathBuilder pathBuilder(*s0);
	DataPrinter printer(&pathBuilder);

	string line;
	// cout << "BSG environment ready. Commands: -A (list actions), -T <id> (take action), -Q (quit)" << endl;
	while(true){
		if(!std::getline(cin, line)) break; // EOF
		if(line.size()==0) continue;

		// parse
		std::stringstream ss(line);
		string cmd;
		ss >> cmd;
		if(cmd=="-Q" || cmd=="quit" || cmd=="exit") break;

		if (cmd == "-A") {
			printer.printActions(vcs, w);
		} else if (cmd == "-B") {
			printer.printBlocks();
		} else if (cmd == "-P") {
			printer.printPlaced();
		} else if (cmd == "-V") {
			printer.printVolume();
		} else if(cmd=="-T") {
			int bid;
			if(!(ss >> bid)){
				cerr << "Usage: -T <block_id>" << endl;
				continue;
			}

			// gather actions and find matching block id
			const clpState& s = pathBuilder.getState();
			list<Action*> actions;
			s.get_actions(actions);
			Action* selected = nullptr;
			for(auto a : actions){
				clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
				if(ca && ca->block.id == bid){ selected = a; break; }
			}

			if(!selected){
				cerr << "No action found for block id " << bid << endl;
				//for(auto a : actions) delete a;
				continue;
			}

			// apply transition
			pathBuilder.addAction(dynamic_cast<clp::clpAction*>(selected->clone()));
			clpAction* selAction = dynamic_cast<clpAction*>(selected); 
			cout << selAction->block.getOccupiedVolume() / s.cont->getVolume() << endl;
		} else {
			cerr << "Unknown command: " << cmd << endl;
		}
	}
}