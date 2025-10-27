#include <iostream>
#include <fstream>
#include "args.hxx"
#include "clpState.h"
#include "VCS_Function.h"
#include "BSG.h"
#include "Greedy.h"

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
    SearchStrategy *gr = new Greedy(vcs);
    BSG *bsg= new BSG(vcs,*gr, w, 0.0, 0);


	// Esperar comandos por stdin para decidir acciones desde fuera del código.
	// Comandos soportados:
	// -A            : imprime las w mejores acciones (una por línea: block_id metrics...)
	// -T <block_id> : aplica la acción cuyo bloque tiene id == block_id y actualiza s0
	// -Q            : salir

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

		if(cmd=="-A"){
			// collect candidate actions
			list<Action*> actions;
			s0->get_actions(actions);

			// evaluate each action using the evaluator
			vector<pair<double, Action*>> scored;
			for(auto a : actions){
				// clear any previous metrics
				dynamic_cast<clp::clpAction*>(a)->metrics.clear();
				double val = vcs->eval_action(*s0, *a);
				scored.push_back(make_pair(val, a));
			}

			// sort descending by score
			sort(scored.begin(), scored.end(), [](const pair<double, Action*>& A, const pair<double, Action*>& B){
				return A.first > B.first;
			});

			// print top w
			int printed=0;
			for(auto &p : scored){
				if(printed>=w) break;
				Action* a = p.second;
				clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
				if(!ca) continue;
				cout << ca->block.id;
				// print metrics
				for(auto m : ca->metrics) cout << " " << m;
				cout << "\n";
				printed++;
			}

			cout << "END" << endl;

			// cleanup actions
			for(auto &p : scored) delete p.second;
		}
		else if(cmd=="-T"){
			int bid;
			if(!(ss >> bid)){
				cerr << "Usage: -T <block_id>" << endl;
				continue;
			}

			// gather actions and find matching block id
			list<Action*> actions;
			s0->get_actions(actions);
			Action* selected = nullptr;
			for(auto a : actions){
				clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
				if(ca && ca->block.id == bid){ selected = a; break; }
			}

			if(!selected){
				cerr << "No action found for block id " << bid << endl;
				for(auto a : actions) delete a;
				continue;
			}

			// apply transition
			s0->transition(*selected);
			// cout << "Applied action for block " << bid << ", new occupied volume: " << s0->cont->getOccupiedVolume() << endl;

			// free remaining actions
			for(auto a : actions) if(a!=selected) delete a;
			delete selected;

			// update evaluator and strategy internal state if needed
			vcs->set_lambda2(vcs->get_lambda2()); // no-op but placeholder for future updates
			bsg->initialize(s0);

		} else if(cmd=="-V"){
			double eval[2] = {(double) s0->cont->getOccupiedVolume(), s0->cont->getVolume()};
			for(auto m : eval) cout << " " << m;
			cout << endl;
		} else {
			cerr << "Unknown command: " << cmd << endl;
		}
	}
}