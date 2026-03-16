#include <iostream>
#include <fstream>
#include "args.hxx"
#include "clpState.h"
#include "VCS_Function.h"
#include "BSG.h"
#include "Greedy.h"
#include "BlockMetrics.h"
#include "DataPrinter.h"

using namespace std;
using namespace metasolver;

// define global TRACE flag used by some modules
bool metasolver::global::TRACE = false;

int main(int argc, char **argv)
{
	args::ArgumentParser parser("********* BSG-ENV *********.", "BSG Environment for CLP.");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::ValueFlag<int> _inst(parser, "int", "Instance", {'i'});
	args::ValueFlag<int> _w(parser, "int", "Beam width (nodes per level)", {'w'});
	args::Flag bsm(parser, "flag", "Use BSM", {"bsm"});
	args::Flag fsb(parser, "fsb", "full-support blocks", {"fsb"});
	args::Positional<std::string> _file(parser, "instance-set", "The name of the instance set");

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help &)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}
	catch (args::ValidationError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	string file = _file.Get();
	int inst = (_inst) ? _inst.Get() : 0;
	double min_fr = 0.98;
	int w = (_w) ? _w.Get() : 4;

	double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
	int seed = 1;

	double r = 0.0;

	Block::FSB = fsb;
	clpState::Format f = clpState::BR;

	clpState *s0 = new_state(file, inst, min_fr, 10000, f);

	VCS_Function *vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);
	SearchStrategy *gr = new Greedy(vcs);

	clpState *best_state = s0;
	double best_volume = 0;

	// --bsm
	BSG *bsg;
	std::list<clpState *> current_nodes;

	// no --bsm
	clpState *current_state;

	if (bsm)
	{
		bsg = new BSG(vcs, *gr, w);
		current_nodes.push_back(s0);
	}
	else
	{
		current_state = s0;
	}

	string line;

	while (true)
	{
		if (!std::getline(cin, line))
			break; // EOF
		if (line.size() == 0)
			continue;

		// parse
		std::stringstream ss(line);
		string cmd;
		ss >> cmd;
		if (cmd == "-Q" || cmd == "quit" || cmd == "exit")
			break;

		if (bsm)
		{
			if (cmd == "-A")
			{
				for (auto n : current_nodes)
				{
					DataPrinter printer(n);
					printer.printActions(vcs, w);
					std::cout << "END" << endl;
				}
			}
			else if (cmd == "-B")
			{
				DataPrinter printer(current_nodes.front());
				printer.printBlocks();
				std::cout << "END" << endl;
			}
			else if (cmd == "-P")
			{
				for (auto n : current_nodes)
				{
					DataPrinter printer(n);
					printer.printPlaced();
					std::cout << "END" << endl;
				}
			}
			else if (cmd == "-S")
			{
				for (auto n : current_nodes)
				{
					DataPrinter printer(n);
					printer.printSpace();
					std::cout << "END" << endl;
				}
			}
			else if (cmd == "-V")
			{
				DataPrinter printer(best_state);
				printer.printVolume();
				std::cout << "END" << endl;
			}
			else if (cmd == "-T")
			{
				std::string arg;
				if (!(ss >> arg))
				{
					std::cerr << "Usage: -T ID1,ID2,...;ID1,ID2,...;..." << std::endl;
					continue;
				}

				std::vector<std::vector<int>> listas;

				std::stringstream ss_listas(arg);
				std::string lista_str;

				while (std::getline(ss_listas, lista_str, ';'))
				{
					std::vector<int> ids;
					std::stringstream ss_ids(lista_str);
					std::string id_str;

					while (std::getline(ss_ids, id_str, ','))
					{
						try
						{
							ids.push_back(std::stoi(id_str));
						}
						catch (...)
						{
							std::cerr << "Invalid ID: " << id_str << std::endl;
							ids.clear();
							break;
						}
					}

					if (!ids.empty())
						listas.push_back(ids);
				}

				map<double, pair<State *, State *>> state_actions;
				int i = 0;
				for (auto s : current_nodes)
				{
					for (auto id : listas[i])
					{
						list<Action *> actions;
						s->get_actions(actions);
						Action *selected = nullptr;
						for (auto a : actions)
						{
							clp::clpAction *ca = dynamic_cast<clp::clpAction *>(a);
							if (ca && ca->block.id == id)
							{
								selected = a;
								break;
							}
						}
						if (selected)
						{
							State *state_copy = s->clone();
							state_copy->transition(*selected);
							double value = gr->run(*state_copy);
							//cout << value << endl;
							state_actions[value] = make_pair(s, state_copy);

							if (value > best_volume)
							{
								best_volume = value;
								best_state = dynamic_cast<clp::clpState *>(state_copy->clone());
							}
						}
					}
					i++;
				}

				list<State *> l = bsg->get_next_states(state_actions);
				current_nodes.clear();
				for (State *s : l)
				{
					clpState* s_copy = static_cast<clpState *>(s);
					list<Action *> actions;
					s_copy->get_actions(actions);
					if (actions.size() > 0) {
						current_nodes.push_back(static_cast<clpState *>(s));
					}
				}
				std::cout << current_nodes.size() << endl;
				//std::cout << "END" << endl;
			}
			else
			{
				cerr << "Unknown command: " << cmd << endl;
			}
		}
		// No --bsm
		else
		{
			DataPrinter printer(current_state);
			if (cmd == "-A")
			{
				printer.printActions(vcs, w);
				std::cout << "END" << endl;
			}
			else if (cmd == "-B")
			{
				printer.printBlocks();
				std::cout << "END" << endl;
			}
			else if (cmd == "-P")
			{
				printer.printPlaced();
				std::cout << "END" << endl;
			}
			else if (cmd == "-S")
			{
				printer.printSpace();
				std::cout << "END" << endl;
			}
			else if (cmd == "-V")
			{
				DataPrinter printer_best(best_state);
				printer_best.printVolume();
				std::cout << "END" << endl;
			}
			else if (cmd == "-T")
			{
				int bid;
				if (!(ss >> bid))
				{
					cerr << "Usage: -T <block_id>" << endl;
					continue;
				}

				// gather actions and find matching block id
				list<Action *> actions;
				current_state->get_actions(actions);
				Action *selected = nullptr;
				for (auto a : actions)
				{
					clp::clpAction *ca = dynamic_cast<clp::clpAction *>(a);
					if (ca && ca->block.id == bid)
					{
						selected = a;
						break;
					}
				}

				if (!selected)
				{
					cerr << "No action found for block id " << bid << endl;
					// for(auto a : actions) delete a;
					continue;
				}

				// apply transition
				current_state->transition(*selected);
				clpAction *selAction = dynamic_cast<clpAction *>(selected);
				std::cout << selAction->block.getOccupiedVolume() << endl;
			}
			else
			{
				cerr << "Unknown command: " << cmd << endl;
			}
		}
	}
}