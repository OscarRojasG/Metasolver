#include <iostream>
#include <fstream>
#include "args.hxx"
#include "clpState.h"
#include "VCS_Function.h"
#include "BSG.h"
#include "Greedy.h"
#include "BlockMetrics.h"
#include "PathBuilder.h"

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
	clpState* s00 = dynamic_cast<clpState*> (s0->clone());

    VCS_Function* vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);
    SearchStrategy *gr = new Greedy(vcs);
    BSG *bsg= new BSG(vcs,*gr, w, 0.0, 0);

	PathBuilder pathBuilder = PathBuilder(*s0);

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
			const clpState& s = pathBuilder.getState();
			s.get_actions(actions);

			// evaluate each action using the evaluator
			vector<pair<double, Action*>> scored;
			for(auto a : actions){
				// clear any previous metrics
				dynamic_cast<clp::clpAction*>(a)->metrics.clear();
				double val = vcs->eval_action(s, *a);
				scored.push_back(make_pair(val, a));
			}

			// sort descending by score
			sort(scored.begin(), scored.end(), [](const pair<double, Action*>& A, const pair<double, Action*>& B){
				return A.first > B.first;
			});

			// print top w
			int printed=0;
			for(auto &p : scored){
				if(printed>=w*w) break;
				Action* a = p.second;
				clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
				if(!ca) continue;
				cout << ca->block.id;
				// print metrics
				for(auto m : ca->metrics) cout << " " << m;
				cout << endl;
				printed++;
			}
			cout << "END" << endl;

			// cleanup actions
			// for(auto &p : scored) delete p.second;
		}
		else if(cmd=="-T"){
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

			// free remaining actions
			//for(auto a : actions) if(a!=selected) delete a;
			// delete selected;
		} else if(cmd=="-B"){
			for (const Block* block:s00->valid_blocks){
				BlockMetrics blockMetrics = BlockMetrics(*block, *(s00->cont));
				cout << block->id;
				cout << " " << blockMetrics.getNormL();
				cout << " " << blockMetrics.getNormH();
				cout << " " << blockMetrics.getNormW();
				cout << " " << blockMetrics.getNormOccupiedVolumeCont();
				cout << " " << blockMetrics.getBoxesAmountReciprocal();
				cout << endl;
			}
			cout << "END" << endl;
		} else if(cmd=="-P"){
			const clpState& s = pathBuilder.getState();
			clp::Space space = s.cont->spaces->top();
			const bool* anchors = space.get_anchor();

			list<const clpAction*> actions = pathBuilder.getActions();
			for (const clpAction* action: actions) {
				const Space& sb = action->space;
				const Block& block = action->block;

				Vector3 coords = sb.get_location(block);

				long bx = anchors[0] ? (coords.getX() + block.getL()) * -1 : coords.getX();
				long by = anchors[1] ? (coords.getY() + block.getW()) * -1 : coords.getY();
				long bz = anchors[2] ? (coords.getZ() + block.getH()) * -1 : coords.getZ();

				cout << block.id;
				cout << " " << bx;
				cout << " " << by;
				cout << " " << bz;
				cout << endl;
			}
			cout << "END" << endl;
		} else if(cmd=="-C"){
			const clpState& s = pathBuilder.getState();
			clp::Space space = s.cont->spaces->top();
			const Vector3 corner = space.get_corner();
			const bool* anchors = space.get_anchor();

			long sx = anchors[0] ? corner.getX() * -1 : corner.getX();
			long sy = anchors[1] ? corner.getY() * -1 : corner.getY();
			long sz = anchors[2] ? corner.getZ() * -1 : corner.getZ();

			cout << " " << sx;
			cout << " " << sy;
			cout << " " << sz;
			cout << endl;
		} else if(cmd=="-V"){
			clpState s = pathBuilder.getState();
			cout << s.cont->getOccupiedVolume() / s.cont->getVolume() << endl;
		} else {
			cerr << "Unknown command: " << cmd << endl;
		}
	}
}