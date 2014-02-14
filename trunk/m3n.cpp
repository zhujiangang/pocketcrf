#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <time.h>
#include <ctime>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "fun.h"
#include "m3n.h"
/* dual problem:
	min_a 1/2 C||\sum_{X,Y} a_X(Y)Df(X,Y)||_2^2 - \sum_{X,Y} a_X(Y) Dt(X,Y)
	s.t.	\sum_Y a_X(Y) = 1
			a_X(Y)>=0,      \forall X,Y
*/

const int MAXSTRLEN=8192;
const int PAGESIZE=8192;
const double EPS=1e-12;
using namespace std;

int int_cmp(const void *a,const void *b){
	return *(int *)a - *(int *)b; 
}


M3N::M3N(){
	_eta=0.000001;
	_C=1;
	_freq_thresh=0;
	_order=0;
	_nbest=1;
	_margin=0;
	_tag_str.set_size(PAGESIZE);
	_x_str.set_size(PAGESIZE);
	_nodes.set_size(PAGESIZE);
	_cliques.set_size(PAGESIZE);
	_clique_node.set_size(PAGESIZE);
	_node_clique.set_size(PAGESIZE);
	_clique_feature.set_size(PAGESIZE);
	_kernel_type=LINEAR_KERNEL;
	_get_kernel=&M3N::linear_kernel;
	_get_kernel_list=&M3N::linear_kernel;
	_kernel_s=1;	// poly_kernel = (_kernel_s*linear_kernel+_kernel_r)^_kernel_d
	_kernel_d=1;	// neural_kernel=tanh(_kernel_s*linear_kernel+_kernel_r)
	_kernel_r=0;	// rbf_kernel = exp(-_kernel_s*norm2)
	_version=12;
	_test_nodes.set_size(PAGESIZE);
	_test_cliques.set_size(PAGESIZE);
	_test_clique_node.set_size(PAGESIZE);
	_test_node_clique.set_size(PAGESIZE);
	_test_clique_feature.set_size(PAGESIZE);

	_mu=NULL;
	_mu_size=0;
	_w=NULL;
	_w_size=0;
	_max_iter=10;
	_obj=0;
	_print_level=0;
}

M3N::~M3N(){
	if(_mu){
		delete [] _mu;
		_mu=NULL;
	}
	if(_w){
		delete [] _w;
		_w=NULL;
	}
}

bool M3N::set_para(char *para_name, char *para_value){
	if(!strcmp(para_name,"C"))
		_C=atof(para_value);
	else if(!strcmp(para_name,"freq_thresh"))
		_freq_thresh=atoi(para_value);
	else if(!strcmp(para_name,"nbest"))
		_nbest=atoi(para_value);
	else if(!strcmp(para_name,"eta"))
		_eta=atof(para_value);
	else if(!strcmp(para_name,"kernel_type")){
		_kernel_type=atoi(para_value);
		if(_kernel_type==LINEAR_KERNEL){
			_get_kernel=&M3N::linear_kernel;
			_get_kernel_list=&M3N::linear_kernel;
		}else if(_kernel_type==POLY_KERNEL){
			_get_kernel=&M3N::poly_kernel;
			_get_kernel_list=&M3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&M3N::neural_kernel;
			_get_kernel_list=&M3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&M3N::rbf_kernel;
			_get_kernel_list=&M3N::rbf_kernel;
		}else{
			cout<<"incorrect kernel_type"<<endl;
			return false;
		}
	}else if(!strcmp(para_name,"kernel_s")){
		_kernel_s=atof(para_value);
	}else if(!strcmp(para_name,"kernel_d")){
		_kernel_d=atoi(para_value);
	}else if(!strcmp(para_name,"kernel_r")){
		_kernel_r=atof(para_value);
	}else if(!strcmp(para_name,"max_iter")){
		_max_iter=atoi(para_value);
	}else if(!strcmp(para_name,"print_level")){
		_print_level=atoi(para_value);
	}else if(!strcmp(para_name,"margin")){
		_margin=atoi(para_value);
	}
		return false;
	return true;
}
bool M3N::learn(char* templet_file, char *training_file,char *model_file, bool relearn){
	int i;
	_model=LEARN_MODEL;
	cout<<"pocket M3N"<<endl<<"version 0."<<_version<<endl<<"Copyright(c)2008 Media Computing and Web Intelligence LAB, Fudan Univ.\nAll rights reserved"<<endl;
	if(relearn){
		if(!load_model(model_file))
			return false;
		_templets.clear();
		_templet_group.clear();
		_path2cliy.clear();
		_gsize=0;
		_order=0;
		_cols=0;
		_ysize=0;
		_path_num=0;
		_node_anum=0;
		_head_offset=0;
		_tags.clear();
		_tag_str.clear();
		_xindex.clear();
		_x_freq.clear();
		_x_str.clear();
		_sequences.clear();
		_nodes.clear();
		_cliques.clear();
		_clique_node.clear();
		_node_clique.clear();
		_clique_feature.clear();
	}
	if(!load_templet(templet_file))
		return false;
	cout<<"templates loaded"<<endl;
	if(!check_training(training_file))
		return false;
	if(!generate_features(training_file))
		return false;
	cout<<"training data loaded"<<endl;
	shrink_feature();
	cout<<"features shrinked"<<endl;
	for(i=0;i<_sequences.size();i++)
		sort_feature(_sequences[i]);
	vector<sequence>(_sequences).swap(_sequences);
	//write model part 1
	initialize();

	write_model(model_file,true);

	_tags.clear();
	_tag_str.clear();
	_xindex.clear();
	_x_freq.clear();
	_x_str.clear();
	_templets.clear();
	
	cout<<"sequence number: "<<_sequences.size()<<endl<<"feature number: "<<_w_size<<endl<<"C: "<<_C<<endl<<"freq_thresh: "<<_freq_thresh<<endl<<"eta: "<<_eta<<endl<<"max_iter: "<<_max_iter<<endl;
	if(_kernel_type==LINEAR_KERNEL)
		cout<<"parameter number: "<<_w_size<<endl;
	else
		cout<<"parameter number: "<<_mu_size<<endl;
	switch(_kernel_type){
		case LINEAR_KERNEL: cout<<"linear kernel: k(a,b)=<a,b>"<<endl;break;
		case POLY_KERNEL: cout<<"polynomial kernel: k(a,b)=("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")^"<<_kernel_d<<endl;break;
		case RBF_KERNEL: cout<<"rbf kernel: k(a,b)=exp{-"<<_kernel_s<<"*||a-b||^2}"<<endl;break;
		case NEURAL_KERNEL: cout<<"neural kernel: k(a,b)=tanh("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")"<<endl;break;
	}
	
	int max_seq_len=0;
	for(i=0;i<_sequences.size();i++){
		if(max_seq_len<_sequences[i].node_num)
			max_seq_len=_sequences[i].node_num;
	}
	_alpha_lattice.resize(max_seq_len*_path_num);
	_v_lattice.resize(max_seq_len*_path_num);
	_optimum_alpha_lattice.resize(max_seq_len*_path_num);
	_optimum_alpha_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_alpha_paths.size();i++)
		_optimum_alpha_paths[i].resize(max_seq_len);
	_optimum_v_lattice.resize(max_seq_len*_path_num);
	_optimum_v_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_v_paths.size();i++)
		_optimum_v_paths[i].resize(max_seq_len);
	int iter=0;
	double kkt_violation;
	double diff=1;
	int converge=0;
	clock_t start_time=clock();
	for(iter=0;iter<_max_iter;){
		//pass through sequences
		double old_obj=_obj;
		for(i=0;i<_sequences.size();i++){
			build_alpha_lattice(_sequences[i]);
			build_v_lattice(_sequences[i]);
			if(find_violation(_sequences[i],kkt_violation))
				smo_optimize(_sequences[i]);
			if(_print_level>0)
				printf("\tseq: %d kkt_violation: %lf\n",i,kkt_violation);
		}
		
		if(iter)
			diff=fabs((old_obj-_obj)/old_obj);
		printf("iter: %d diff: %lf obj: %lf\n",iter,diff,_obj);
		iter++;
		if(diff<_eta)
			converge++;
		else
			converge=0;
		if(converge==3)
			break;
	}
	double elapse = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
	cout<<"elapse: "<<elapse<<" s"<<endl;
	//write model part 2
	write_model(model_file,false);
	return true;
}
bool M3N::write_model(char *model_file, bool first_part){
	ofstream fout;
	if(first_part){
		char tmp_model_file[1000];
		sprintf(tmp_model_file,"%s.tmp",model_file);
		fout.open(tmp_model_file);
		if(!fout.is_open()){
			cout<<"can not open model file: "<<model_file<<endl;
			return false;
		}
		int i,j,k,ii;
		//write version
		fout<<"version\t"<<_version<<endl;
		//write kernel_type
		fout<<_kernel_type<<endl;
		//write kernel parameters
		fout<<_kernel_s<<endl;
		fout<<_kernel_d<<endl;
		fout<<_kernel_r<<endl;

		//write eta
		fout<<_eta<<endl;
		//write C
		fout<<_C<<endl;
		//write freq_thresh
		fout<<_freq_thresh<<endl;

		//write templets
		for(i=0;i<_templets.size();i++){
			templet &cur_templet=_templets[i];
			for(j=0;j<cur_templet.x.size();j++)
				fout<<cur_templet.words[j].c_str()<<"%x["<<cur_templet.x[j].first<<","<<cur_templet.x[j].second<<"]";
			fout<<cur_templet.words[j].c_str();
			for(j=0;j<cur_templet.y.size();j++)
				fout<<"%y["<<cur_templet.y[j]<<"]";
			fout<<endl;
		}
		fout<<endl;
		//write y
		fout<<_ysize<<endl;
		for(i=0;i<_tags.size();i++)
			fout<<_tags[i]<<endl;
		fout<<endl;
		//write x
		fout<<_cols<<endl<<endl;
		fout<<_xindex.size()<<endl;
		map<char*, int, str_cmp>::iterator it;
		for(it = _xindex.begin(); it != _xindex.end(); it++)
			fout<<it->first<<"\t"<<it->second<<endl;
		fout<<endl;
		if(_kernel_type!=LINEAR_KERNEL){
			//write _sequences info
			fout<<_sequences.size()<<endl;//sequence number
			for(i=0;i<_sequences.size();i++){
				sequence &seq=_sequences[i];
				fout<<seq.node_num<<"\t"<<seq.mu_index<<endl;//sequence node_num, mu_index
				for(j=0;j<seq.node_num;j++){
					node &nod=seq.nodes[j];
					fout<<nod.key<<"\t"<<nod.clique_num<<endl;//node key, clique num
					//for each clique, 	output groupid, feature_num, node_num, key, fvector
					for(k=0;k<nod.clique_num;k++){
						if(!nod.cliques[k])
							continue;
						clique &cli=*(nod.cliques[k]);
						fout<<cli.groupid<<"\t"<<cli.feature_num<<"\t"<<cli.node_num<<"\t"<<cli.key<<endl;
						for(ii=0;ii<cli.feature_num-1;ii++)
							fout<<cli.fvector[ii]<<"\t";
						fout<<cli.fvector[ii]<<endl;
					}
				}
			}
			fout<<endl;
		}
		fout.close();
	}else{
		//copy tmp_model_file
		char tmp_model_file[1000];
		char line[MAXSTRLEN];
		sprintf(tmp_model_file,"%s.tmp",model_file);
		ifstream fin;
		fin.open(tmp_model_file);
		fout.open(model_file);

		while(true){
			fin.getline(line,MAXSTRLEN-1);
			if(!fin.eof())
				fout<<line<<endl;
			else
				break;
		}
		fin.close();
		unlink(tmp_model_file);
		
		//write obj
		int i;
		
		fout<<_obj<<endl;
		if(_kernel_type==LINEAR_KERNEL){
			//write w
			fout<<_w_size<<endl;
			for(i=0;i<_w_size;i++){
				fout<<_w[i]<<endl;
			}
			fout<<endl;
		}
		//write mu
		fout<<_mu_size<<endl;
		for(i=0;i<_mu_size;i++){
			fout<<_mu[i]<<endl;	
		}
		fout.close();
	}
	return true;
}


bool M3N::add_templet(char *line ,int &cur_group){
	if(!line[0]||line[0]=='#') 
		return false;

	templet n;
	char *p=line,*q;
	char word[1000];
	

	char index_str[1000];
	int index1,index2;
	while(q=catch_string(p,"%x[",word)){
		p=q;
		n.words.push_back(word);
		p=catch_string(p,",",index_str);
		index1=atoi(index_str);
		p=catch_string(p,"]",index_str);
		index2=atoi(index_str);
		n.x.push_back(make_pair(index1,index2));
	}
	q=catch_string(p,"%y[",word);
	if(!q){
		cout<<"templet: "<<line<<" incorrect"<<endl;
		return false;
	}
	n.words.push_back(word);

	p=q-3;
	while(p=catch_string(p,"%y[","]",index_str)){
		index1=atoi(index_str);
		n.y.push_back(index1);
	}
	if(_templets.size()){//check end_of_group
		templet &last_n=_templets.back();
		int i;
		for(i=0;i<last_n.y.size();i++){
			if(last_n.y[i]!=n.y[i]){//start of group
				last_n.end_of_group=true;
				n.groupid=++cur_group;
				if(-n.y[0]>_order)		_order=-n.y[0];
				break;
			}
		}
		if(i==last_n.y.size()){//not start of group
			n.groupid=cur_group;
			last_n.end_of_group=false;
		}
	}else{//first templet, start of group
		_order=-n.y[0];
		n.groupid=cur_group=0;
		n.end_of_group=false;
	}
	_templets.push_back(n);
	return true;
}

bool M3N::load_templet(char *templet_file){
	ifstream fin;
	//read template
	fin.open(templet_file);
	if (!fin.is_open()){
		cout<<"template file: "<<templet_file<<" not found"<<endl;
		return false;
	}
	char line[MAXSTRLEN];
	int cur_group;
	while(!fin.eof()){
		fin.getline(line,MAXSTRLEN-1);
		add_templet(line,cur_group);
	}
	fin.close();
	_templets.back().end_of_group=true;
	_gsize=cur_group+1;
	_templet_group.resize(_gsize);
	
	if(_order<0)// no _templets
		return false;
	return true;
}

bool M3N::check_training(char *training_file){
	ifstream fin;
	fin.open(training_file);
	if(!fin.is_open())	return false;
	char line[MAXSTRLEN];
	int lines=0;
	_cols=0;
	while(!fin.eof()){//fgets(line,MAXSTRLEN-1,fp))
		fin.getline(line,MAXSTRLEN-1);
		lines++;
		if(!line[0]) continue;
		vector<char *>columns;
		if(!split_string(line,"\t",columns)){
			cout<<"columns should be greater than 1"<<endl;
			fin.close();
			return false;//columns should be greater than 1
		}
		if(_cols && _cols!=columns.size()){//incompatible
			cout<<"line: "<<lines<<" columns incompatible"<<endl;
			fin.close();
			return false;
		}
		_cols=columns.size();

		char *t=columns.back();//tag
		int index;
		if(!vector_search(_tags,t,index,str_cmp())){
			char *p=_tag_str.push_back(t);//copy string
			vector_insert(_tags,p,index);
		}
	}
	fin.close();
	_ysize=_tags.size();
	set_group();
	return true;
}

void M3N::set_group(){
	//calculate _templet_group
	//set the size of each group
	//_order and _tags.size() must be known
	int i,j,k;
	for(i=0,j=0;i<_templets.size();i++){
		if(_templets[i].end_of_group){
			int n=(int)pow((double)_ysize,(int)_templets[i].y.size());
			_templet_group[j++].resize(n);//group j has n offsets
		}
	}
	vector<int> path_index(_order+1,0);
	int path_size=(int)pow((double)_ysize,_order+1);
	for(i=0;i<path_size;i++){//assosiate path i with _templet_group
		int cur_group=0;
		for(j=0;j<_templets.size();j++){
			if(_templets[j].end_of_group){
				vector<int> &ytemp=_templets[j].y;
				int offset;
				vector<int> temp;
				for(k=0,offset=0;k<ytemp.size();k++)
					offset=offset*_ysize+path_index[-ytemp[k]];
				_templet_group[cur_group++][offset].push_back(i);//path i added to current group's offset
			}
		}
		for(j=0;j<_order+1 && path_index[j]==_ysize-1;j++);
		if(j==_order+1) break;
		path_index[j]++;
		for(j--;j>=0;j--)	path_index[j]=0;
	}
	for(i=0;i<_templet_group.size();i++)
		for(j=0;j<_templet_group[i].size();j++)
			((vector<int>)(_templet_group[i][j])).swap(_templet_group[i][j]);

	_path_num=1;
	for(i=0;i<_order+1;i++)
		_path_num*=_ysize;
	_path2cliy.resize(_templet_group.size());
	for(i=0;i<_path2cliy.size();i++)
		_path2cliy[i].resize(_path_num,-1);
	for(i=0;i<_templet_group.size();i++){
		for(j=0;j<_templet_group[i].size();j++)
			for(k=0;k<_templet_group[i][j].size();k++)
				_path2cliy[i][_templet_group[i][j][k]]=j;
	}
}

bool M3N::generate_features(char *training_file){
	char line[MAXSTRLEN];
	vector<char *>table;// table[i,j] = table[i*_cols+j]
	charlist table_str;
	table_str.set_size(PAGESIZE);//1 page memory
	_w_size=0;//lambda size
	int lines=0;
	int i;
	ifstream fin;
	fin.open(training_file);
	while(!fin.eof()){
		fin.getline(line,MAXSTRLEN-1);
		if(line[0]){
			vector<char *>columns;
			split_string(line,"\t",columns);
			for(i=0;i<_cols;i++){
				char *p=table_str.push_back(columns[i]);
				table.push_back(p);
			}
		}else if(table.size()){
			add_x(table);
			table.clear();//prepare for new table
			table_str.free();
			lines++;
			if(!(lines%100))
				printf("%d.. ",lines);
		}
	}
	if(table.size()){//non-empty line
		add_x(table);
		table.clear();//prepare for new table
		table_str.free();
		lines++;
		if(!(lines%100))
			printf("%d.. ",lines);
	}
	fin.close();
	return true;
}

bool M3N::add_x(vector<char *> &table){
	int i,j,k,c;
	vector<int> y;
	int rows=table.size()/_cols;
	char s[1024];
	char s1[1024];
	char s2[1024];
	sequence seq;
	node* nod=_nodes.alloc(rows);//
	seq.node_num=rows;
	seq.nodes=nod;

	_sequences.push_back(seq);
	for(i=0;i<rows;i++){
		y.resize(y.size()+1);
		vector_search(_tags,table[(i+1)*_cols-1],y.back(),str_cmp());//get the tag of current node
		nod[i].key=0;
		for(j=i-_order;j<=i;j++)
			if(j>=0)
				nod[i].key=nod[i].key*_ysize+y[j];

		vector<clique*> clisp;//features that affect on current nodes
		int cur_group=0;
		vector<int> feature_vector;
		for(j=0;j<_templets.size();j++){
			//get first y's offset
			templet &pat=_templets[j];
			if(pat.y[0]+i<0)
				continue;
			//get x, here s=x
			sprintf(s, "%d", j);
			strcat(s,":");
			int index1,index2;
			for(k=0;k<pat.x.size();k++){
				strcat(s,pat.words[k].c_str());
				strcat(s,"//");
				index1=pat.x[k].first+i;
				index2=pat.x[k].second;
				assert(index2>=0 && index2<_cols-1);
				if(index1<0){
					index1=-index1-1;
					strcpy(s1,"B_");
					sprintf(s2,"%d",index1);
					strcat(s1,s2);//B_0 for example
				}else if(index1>=rows){
					index1-=rows;
					strcpy(s1,"E_");
					sprintf(s2,"%d",index1);
					strcat(s1,s2);//E_0 for example
				}else if(!strlen(table[index1*_cols+index2])){
					s[0]=0;//null feature
					break;
				}else{
					strcpy(s1,table[index1*_cols+index2]);
				}
				strcat(s,s1);
				strcat(s,"//");
			}
			if(s[0]){
				strcat(s,pat.words[k].c_str());
				//x obtained, insert x
				int index;//index of feature s

				if(insert_x(s,index)){
					c=pow((double)_ysize,(int)pat.y.size());
					_w_size+=c;
				}
				//get clique
				feature_vector.push_back(index);
			}
			if(pat.end_of_group)
			{//creat new clique
				clique cli;
				vector<node*> ns;
				int key=0;
				for(k=0;k<pat.y.size();k++){
					ns.push_back(nod+i+pat.y[k]);
					key=key*_ysize+ y[i+pat.y[k]];
				}
				node ** np=_clique_node.push_back(&ns[0],ns.size());
				cli.nodes=np;
				cli.node_num=ns.size();
				cli.key=key;
				if(feature_vector.size())
					cli.fvector=_clique_feature.push_back(&feature_vector[0],feature_vector.size());
				else
					cli.fvector=NULL;
				cli.feature_num=feature_vector.size();
				cli.groupid=_templets[j].groupid;
				clique *new_clique=_cliques.push_back(&cli,1);
				clisp.push_back(new_clique);
				feature_vector.clear();
			}
		}
		//set node -> clique
		if(clisp.size())
			nod[i].cliques = _node_clique.push_back(&clisp[0],clisp.size());
		else
			nod[i].cliques=NULL;
		nod[i].clique_num =clisp.size();
	}
	return true;
}

bool M3N::insert_x(char *target, int &index){
	map<char *, int , str_cmp>::iterator p;
	p=_xindex.find(target);
	if(p!=_xindex.end()){
		index=p->second;
		_x_freq[index]++;
		return false;
	}else{
		char *q=_x_str.push_back(target);
		_xindex.insert(make_pair(q,_w_size));
		index=_w_size;
		_x_freq.resize(index+1);
		_x_freq[index]=1;
		return true;
	}
}

void M3N::shrink_feature(){
	if(_freq_thresh<=1)
		return;
    map<int, int> old2new;
    int new_lambda_size = 0;
	char temp[MAXSTRLEN];
	map<char*, int, str_cmp>::iterator it;
    for(it=_xindex.begin(); it!= _xindex.end();){
		if (_x_freq[it->second] >=_freq_thresh)
		{
			char *key=it->first;
			catch_string(key,":",temp);
			int index=atoi(temp);
			int gram_num=_templets[index].y.size();
			old2new.insert(make_pair<int, int>(it->second, new_lambda_size));
			it->second = new_lambda_size;
			new_lambda_size += (int)pow(double(_ysize),gram_num);
			++it;
		}else{
			_xindex.erase(it++);
		}
    }	
	_w_size=new_lambda_size;
	map<int, int>::iterator iter;
	freelist<int> temp_clique_feature;
	temp_clique_feature.set_size(PAGESIZE*16);
	_clique_feature.free();
	int i,j,k,ii;
	for(i=0;i<_sequences.size();i++)
	{
		sequence &seq=_sequences[i];
		for(j=0;j<seq.node_num;j++)
		{
			node &nod=seq.nodes[j];
			for(k=0;k<nod.clique_num;k++)
			{
				if(!nod.cliques[k])
					continue;
				clique &cli=*nod.cliques[k];
				vector<int> newf;
				for(ii=0;ii<cli.feature_num;ii++)
				{
					iter = old2new.find(cli.fvector[ii]);
					if(iter != old2new.end())
						newf.push_back(iter->second);
				}
				int *f;
				if(newf.size())
					f=temp_clique_feature.push_back(&newf[0],newf.size());
				else
					f=NULL;
				cli.fvector=f;
				cli.feature_num=newf.size();
			}
		}
	}
	_clique_feature.clear();
	for(i=0;i<_sequences.size();i++)
	{
		sequence &seq=_sequences[i];
		for(j=0;j<seq.node_num;j++)
		{
			node &nod=seq.nodes[j];
			for(k=0;k<nod.clique_num;k++)
			{
				if(!nod.cliques[k])
					continue;
				clique &cli=*nod.cliques[k];
				if(cli.feature_num)
				{	
					int *f=_clique_feature.push_back(cli.fvector,cli.feature_num);
					cli.fvector=f;
				}else{
					cli.fvector=NULL;
				}
			}
		}
	}
    return;
}

void M3N::initialize(){
	int i,j;
	//set key path for all nodes
	_node_anum=pow((double)_ysize,_order);
	_path_num=_node_anum*_ysize;
	if(_kernel_type==LINEAR_KERNEL){
		if(!_w){//if _w != NULL => _w has been initalized with model file
			_w=new double[_w_size];
			memset(_w,0,sizeof(double)*_w_size);
		}
	}
	//initialize mu
	if(!_mu){
		vector<double> mu;// marginal probabilities for all paths
		for(i=0;i<_sequences.size();i++){
			_sequences[i].mu_index=mu.size();
			for(j=0;j<_sequences[i].node_num;j++){
				int mu_size=mu.size();
				mu.resize(mu_size+_path_num,0);
				mu[mu_size+_sequences[i].nodes[j].key]=1;
			}
		}
		_mu_size=mu.size();
		_mu=new double[_mu_size];
		memcpy(_mu,&mu[0],sizeof(double)*_mu_size);
	}else{
		//relearn: assign seq.mu_index
		int mu_size=0;
		for(i=0;i<_sequences.size();i++){
			_sequences[i].mu_index=mu_size;
			mu_size+=_sequences[i].node_num*_path_num;
		}
	}
	_head_offset=-log((double)_ysize)*_order;
	_clique_kernel.resize(_gsize);
	_path_kernel.resize(_path_num*_path_num);
}

bool M3N::find_violation(sequence &seq, double &kkt_violation){
	int i,j;
	kkt_violation=0;

	forward_backward_viterbi(_ysize,_order,seq.node_num,_alpha_lattice,_optimum_alpha_lattice,_optimum_alpha_paths,_head_offset,LOGZERO,LOGZERO,INF);
	forward_backward_viterbi(_ysize,_order,seq.node_num,_v_lattice,_optimum_v_lattice,_optimum_v_paths,0,0,-INF,INF);
	
	vector<int> v_index;
	vector<double> vyc;
	vector<int> a_index;
	vector<double> ayc;
	double v_yc;
	int v_yc_index;
	double a_yc;
	int a_yc_index;
	int y1=-1;
	int y2;
	int violation=0;
	//check
	
	for(i=0;i<seq.node_num;i++){
		
		//order v,alpha
		v_index.clear();
		vyc.clear();
		a_index.clear();
		ayc.clear();
		vyc.assign(_optimum_v_lattice.begin()+i*_path_num,_optimum_v_lattice.begin()+(i+1)*_path_num);
		merge_sort(vyc,v_index);
		ayc.assign(_optimum_alpha_lattice.begin()+i*_path_num,_optimum_alpha_lattice.begin()+(i+1)*_path_num);
		merge_sort(ayc,a_index);


		for(j=0;j<_path_num;j++){
			//find v(~y_c),alpha(~y_c)

			if(v_index[0]!=j){
				v_yc=vyc[v_index[0]];
				v_yc_index=v_index[0];
			}else{
				v_yc=vyc[v_index[1]];
				v_yc_index=v_index[1];
			}

			if(a_index[0]!=j){
				a_yc=ayc[a_index[0]];
				a_yc_index=a_index[0];
			}else{
				a_yc=ayc[a_index[1]];
				a_yc_index=a_index[1];
			}

			if(ayc[j]==LOGZERO && vyc[j]>v_yc+EPS){//violation=1;
				if(kkt_violation<vyc[j]-v_yc){
					kkt_violation=vyc[j]-v_yc;
					y1=i*_path_num+j;
					//assert(a_yc>LOGZERO);
					y2=i*_path_num+a_yc_index;
					_path1=_optimum_v_paths[y1];
					_path2=_optimum_alpha_paths[y2];
				}
			}else if(ayc[j]>LOGZERO && vyc[j]<v_yc-EPS){//violation=2;
				if(kkt_violation<v_yc-vyc[j]){
					kkt_violation=v_yc-vyc[j];
					y1=i*_path_num+j;
					//assert(v_yc>vyc[j]);
					y2=i*_path_num+v_yc_index;
					_path1=_optimum_alpha_paths[y1];
					_path2=_optimum_v_paths[y2];
				}
			}
		}
	}
	if(y1!=-1)
		return true;
	return false;
}
void M3N::build_alpha_lattice(sequence &seq){
	//_alpha_lattice[i*_path_num+j] is the log distribution on the marginal probability
	int i,j;
	memcpy(&_alpha_lattice[0],_mu+seq.mu_index,seq.node_num*_path_num*sizeof(double));
	vector<double> margin(_node_anum);//separator function 
	for(i=0;i<seq.node_num-1;i++){//i th node , j th path
		fill(margin.begin(),margin.end(),0);
		for(j=0;j<_path_num;j++)
			margin[j%_node_anum]+=_alpha_lattice[i*_path_num+j];
		for(j=0;j<_path_num;j++)//if margin>0, _alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]/margin[j%_node_anum]), otherwise, _alpha_lattice[i*_path_num+j]=0
			if(_alpha_lattice[i*_path_num+j]>0)
				_alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]/margin[j%_node_anum]);	//convert to log probability
			else if(margin[j%_node_anum]>0)
				_alpha_lattice[i*_path_num+j]=LOGZERO;//log(0)
			else
				_alpha_lattice[i*_path_num+j]=_head_offset;
	}
	for(j=0;j<_path_num;j++){//i=seq.node_num-1
			if(_alpha_lattice[i*_path_num+j]>0)
				_alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]);
			else
				_alpha_lattice[i*_path_num+j]=LOGZERO;//log(0)
	}
}
void M3N::build_v_lattice(sequence &seq){
	int i,j,k,ii,jj;
	//build _v_lattice
	if(_model==TEST_MODEL && _v_lattice.size()<seq.node_num*_path_num)
		_v_lattice.resize(seq.node_num*_path_num);
	fill(_v_lattice.begin(),_v_lattice.end(),0);
	if(_kernel_type==LINEAR_KERNEL){
		//LEARN_MODEL: v(x,y)=w'f(x,y)+l(x,y)=\sum_t w'f(x,y,t)+l(x,y,t)=\sum_t phi(x,y,t)
		//TEST_MODEL:  v(x,y)=w'f(x,y)       =\sum_t w'f(x,y,t)
		for(i=0;i<seq.node_num;i++){
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			for(j=0;j<n1.clique_num;j++){
				clique &cli=*(n1.cliques[j]);
				vector<vector<int> > &group=_templet_group[cli.groupid];
				for(k=0;k<cli.feature_num;k++){
					for(ii=0;ii<group.size();ii++){
						for(jj=0;jj<group[ii].size();jj++){
							_v_lattice[i*_path_num+group[ii][jj]]+=_w[cli.fvector[k]+ii];
						}
					}
				}
			}
			if(_model==LEARN_MODEL){
				for(j=0;j<_path_num;j++)
					if(j%_ysize!=n1.key%_ysize)
						_v_lattice[i*_path_num+j]++;
			}
		}
	}else{
		//v(x,y)=\sum_t phi(x,y,t)=\sum_t w'f(x,y,t)+l(x,y,t)=\sum_t C*\sum_{x`,y`} a(x`,y`)Df(x`,y`)f(x,y,t) + l(x,y,t)
		//phi(x,y,t)
		//=C*\sum_{x`}\{ \sum_{t`} [ K(x`,y_{x`},t`,x,y,t)-\sum_{y`_{t`}} (  K(x`,y`_{t`},x,y,t)mu(y`_{t`}) ) ] \}
		//  +I(y_t != {y_x}_t)
		//=C*\sum_{x`}\{ \sum_{t`} [ \sum_{q \in x_t, q` \in x`_t`}I({y`_x`_t`}_q`=={y_t}_q) K(q,q`)
		//						-mu(y`_{t`}\sum_{q \in x_t, q` \in x`_t`} I({y`_t`}_q`=={y_t}_q) K(q,q`) ] \} +I(y_t != {y_x}_t)
		//thus, need to calculate kernels K(q,q`) forall clique pair (q,q`)

		//K(x`,y`,t`,x,y,t)=\sum_{q \in x_t, q` \in x`_t`}I({y`_t`}_q`=={y_t}_q) K(q,q`)
		for(i=0;i<seq.node_num;i++){
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			//calculate n1's kernel
			for(j=0;j<_sequences.size();j++){
				if(_print_level>1)
					cout<<"\t\tkernel compute: k(seq["<<i<<"],seq["<<j<<"])"<<endl;
				for(k=0;k<_sequences[j].node_num;k++){
					node &n2=_sequences[j].nodes[k];
					get_kernel(n1,n2);//calculate kernels
					//calculate v
					for(ii=0;ii<_path_num;ii++){
						_v_lattice[i*_path_num+ii]+=_path_kernel[ii*_path_num+n2.key];
						for(jj=0;jj<_path_num;jj++){
							_v_lattice[i*_path_num+ii]-=_path_kernel[ii*_path_num+jj]*_mu[_sequences[j].mu_index+k*_path_num+jj];
						}
					}
				}
			}
			for(j=0;j<_path_num;j++){
				_v_lattice[i*_path_num+j]*=_C;
				if(_model==LEARN_MODEL)
					if(j%_ysize!=n1.key%_ysize)
						_v_lattice[i*_path_num+j]++;
			}
		}
	}
}
void M3N::smo_optimize(sequence &seq){
	
	int i,j,k;
	//calculate alpha1 , alpha2, v1 ,v2
	double alpha1=0;
	double alpha2=0;
	double v1=0;
	double v2=0;
	for(i=0;i<seq.node_num;i++){
		alpha1+=_alpha_lattice[i*_path_num+_path1[i]];
		alpha2+=_alpha_lattice[i*_path_num+_path2[i]];
		v1+=_v_lattice[i*_path_num+_path1[i]];
		v2+=_v_lattice[i*_path_num+_path2[i]];
	}
	if(alpha1>LOGZERO)
		alpha1=exp(alpha1);
	else
		alpha1=0;
	if(alpha2>LOGZERO)
		alpha2=exp(alpha2);
	else
		alpha2=0;
	
	double kern=0;
	if(_kernel_type==LINEAR_KERNEL){// faster: O(node_num) complexity
		//improved by Xuancong Wang
		//||f(x,_path1)-f(x,_path2)||_2^2
		map <int,pair<int,int> > fvec_pair;
		for(int i=0;i<seq.node_num;i++){
			if(_path1[i]==_path2[i])
				continue;

			int J = seq.nodes[i].clique_num;
			clique **cs = seq.nodes[i].cliques;
			for(int j=0; j<J; ++j){
				int K=cs[j]->feature_num;
				int *fv=cs[j]->fvector;
				vector<int> &path2cliy = _path2cliy[cs[j]->groupid];
				for(int k=0; k<K; ++k){
					++fvec_pair[fv[k]+path2cliy[_path1[i]]].first;
					++fvec_pair[fv[k]+path2cliy[_path2[i]]].second;
				}
			}
		}
		for(map <int,pair<int,int> >::iterator it=fvec_pair.begin(); it!=fvec_pair.end();
++it){
			int diff = it->second.first-it->second.second;
			kern += diff*diff;
		}
	}else{  
		int path_list_1[4];
		int path_list_2[4];
		//||f(x,_path1)-f(x,_path2)||_2^2=\sum_{i,j} [K(_path1,i;_path1,j)+K(_path2,i;_path2,j)-K(_path1,i;_path2,j)-K(_path2,i;_path1,j)]
		for(i=0;i<seq.node_num;i++){
			for(j=0;j<seq.node_num;j++){


				path_list_1[0]=_path1[i];
				path_list_1[1]=_path2[i];
				path_list_1[2]=_path1[i];
				path_list_1[3]=_path2[i];

				path_list_2[0]=_path1[j];
				path_list_2[1]=_path2[j];
				path_list_2[2]=_path2[j];
				path_list_2[3]=_path1[j];

				get_kernel(seq.nodes[i],seq.nodes[j],path_list_1,path_list_2,4);
				kern+=_path_kernel[0];
				kern+=_path_kernel[1];
				kern-=_path_kernel[2];
				kern-=_path_kernel[3];
			}
		}
	}
	//delta=max{-alpha(_path1), min{alpha(_path2),delta_v/(_C*||f(x,_path1)-f(x,_path2)||_2^2)}}
	double delta=(v1-v2)/(_C*kern);
	delta=delta<alpha2?delta:alpha2;
	delta=delta>(-alpha1)?delta:(-alpha1);
	//update mu
	//for each mu on _path1
	//	mu+=delta
	//for each mu on _path2
	//	mu-=delta
	for(i=0;i<seq.node_num;i++){
		_mu[i*_path_num+_path1[i]+seq.mu_index]+=delta;
		_mu[i*_path_num+_path2[i]+seq.mu_index]-=delta;
	}
	//update obj
	//obj=1/2 C||\sum_{X,Y} a_X(Y)Df(X,Y)||_2^2 - \sum_{X,Y} a_X(Y) Dt(X,Y)
	//v(x,y)= C*\sum_{x`,y`} a(x`,y`)Df(x`,y`)f(x,y) + l(x,y)
	//=>
	//D obj=-delta[v(x,y1)-v(x,y2)]+1/2 C delta^2 ||f(x,y1)-f(x,y2)||_2^2
	_obj+=-(v1-v2)*delta+0.5*_C*kern*delta*delta;
	if(_kernel_type==LINEAR_KERNEL){
		//update _w
		//_w=C*\sum_{x,y}a(x,y)Df(x,y)=C*\sum_{x,y,t}mu(x,y,t)Df(x,y,t)=C*\sum_{x,y}f(x,y_x)-\sum_{x,y,t}mu(x,y,t)f(x,y,t)
		//=>
		//D _w=-C*\sum_{x,y,t}Dmu(x,y,t)f(x,y,t)=-C*delta*[f(x,y1)+f(x,y2)]
		for(i=0;i<seq.node_num;i++){
			for(j=0;j<seq.nodes[i].clique_num;j++){
				if(!seq.nodes[i].cliques[j])
					continue;
				clique &cli=*(seq.nodes[i].cliques[j]);
				for(k=0;k<cli.feature_num;k++){
						_w[cli.fvector[k]+_path2cliy[cli.groupid][_path1[i]]]-=delta*_C;
						_w[cli.fvector[k]+_path2cliy[cli.groupid][_path2[i]]]+=delta*_C;
				}
			}
		}
	}
}


void M3N::sort_feature(sequence &seq){
//for each node
//	sort clique by clique.group_id, this has been automatically set since cliques are generated in template order
//for each clique
//	sort fvector
	int i,j;

	for(i=0;i<seq.node_num;i++){
		for(j=0;j<seq.nodes[i].clique_num;j++){
			if(!seq.nodes[i].cliques[j]||!seq.nodes[i].cliques[j]->fvector)
				continue;
			qsort(seq.nodes[i].cliques[j]->fvector,seq.nodes[i].cliques[j]->feature_num,sizeof(int),int_cmp);
		}
	}
	
}
void M3N::assign_tag(sequence &seq, vector<int> &node_tag)
{
	int i,j,k;
	for(i=0;i<seq.node_num;i++){
		seq.nodes[i].key=0;
		for(j=0;j<=_order;j++){
			if(i+j>=_order)
				seq.nodes[i].key=seq.nodes[i].key*_ysize+node_tag[i+j-_order];
		}
	}
	for(i=0;i<seq.node_num;i++)
	{
		node &nod=seq.nodes[i];
		for(j=0;j<nod.clique_num;j++)
		{
			if(!nod.cliques[j])
				continue;
			clique &cli=*(nod.cliques[j]);
			int key=0;
			for(k=0;k<cli.node_num;k++)
				key= key*_ysize +cli.nodes[k]->key%_ysize;
			cli.key=key;
		}
	}
}
void M3N::node_margin(sequence &seq, vector<vector<double> >&node_p,vector<double> &alpha, vector<double> &beta, double &z)
{
	int i,j;
	node_p.resize(seq.node_num);
	for(i=0;i<seq.node_num;i++)
		node_p[i].resize(_ysize);
	vector<int> first_cal(_ysize*seq.node_num,1);
	for(i=0;i<seq.node_num;i++)
	{
		int *cur_first=&first_cal[_ysize*i];
		double *cur_p=&node_p[i][0];
		for(j=0;j<_node_anum;j++)
		{
			int index = j % _ysize;
			if(cur_first[index])
			{
				cur_first[index]=0;
				cur_p[index]=alpha[i*_node_anum+j]+beta[i*_node_anum+j];
			}else
				cur_p[index]=log_sum_exp(alpha[i*_node_anum+j]+beta[i*_node_anum+j],cur_p[index]);
		}
		for(j=0;j<_ysize;j++)
			cur_p[j]=exp(cur_p[j]-z);
	}
}
double M3N::path_cost(sequence &seq, vector<double>& lattice){
	int i;
	double c=0;
	for(i=0;i<seq.node_num;i++)
		c+=lattice[_path_num*i+seq.nodes[i].key];
	return c;
}
void M3N::tag(vector<vector<string> > &table, vector<vector<string> > &best_tag){
	vector<double> sequencep;
	vector<vector<double> > nodep;
	tag(table,best_tag);
}
void M3N::tag(vector<vector<string> > &table, vector<vector<string> > &best_tag,vector<double> &sequencep, vector<vector<double> > &nodep){
	_model=TEST_MODEL;
	sequence seq;
	generate_sequence(table,seq);
	sort_feature(seq);
	build_v_lattice(seq);
	vector<vector<int> > best_path;
	viterbi(seq.node_num, _order, _ysize, _nbest, _v_lattice, best_path);
	int i,j;
	for(i=0;i<best_path.size();i++){
		vector<string> cur_tag(seq.node_num);
		for(j=0;j<seq.node_num;j++){
			cur_tag[j]=_tags[best_path[i][j]];
		}
		best_tag.push_back(cur_tag);
	}
	if(_margin){
		vector<double> alpha;
		vector<double> beta;
		double z;
		forward_backward(_ysize,_order,seq.node_num,_v_lattice,alpha,beta,z);
		node_margin(seq,nodep,alpha,beta,z);
		sequencep.resize(best_path.size());
		for(i=0;i<best_path.size();i++)
		{
			assign_tag(seq,best_path[i]);
			double c=path_cost(seq,_v_lattice);
			sequencep[i]=exp(c-z);
		}
	}
}

void M3N::generate_sequence(std::vector<vector<string> > &table, sequence &seq){
	int i,j,k;
	int rows=table.size();
	char s[1024];
	char s1[1024];
	char s2[1024];
	node* nod=_test_nodes.alloc(rows);
	seq.node_num=rows;
	seq.nodes=nod;
	for(i=0;i<rows;i++)
	{
		nod[i].key=0;//random initialize
		vector<clique*> clisp;//features that affect on current nodes
		int cur_group=0;
		vector<int> feature_vector;
		for(j=0;j<_templets.size();j++)
		{
			//get first y's offset
			templet &pat=_templets[j];
			if(pat.y[0]+i<0)
				continue;
			//get x, here s=x
			sprintf(s,"%d",j);
			strcat(s,":");

			int index1,index2;
			for(k=0;k<pat.x.size();k++)
			{
				strcat(s,pat.words[k].c_str());
				strcat(s,"//");
				index1=pat.x[k].first+i;
				index2=pat.x[k].second;
				assert(index2>=0 && index2<_cols-1);//x[index1,index2], index2 should be >=0 and < xcols
				if(index1<0)
				{
					index1=-index1-1;
					strcpy(s1,"B_");
					sprintf(s2,"%d",index1);
					strcat(s1,s2);//B_0 for example
				}else if(index1>=rows){
					index1-=rows;
					strcpy(s1,"E_");
					sprintf(s2,"%d",index1);
					strcat(s1,s2);//E_0 for example
				}else if(!strlen(table[index1][index2].c_str())){
					s[0]=0;
					break;
				}else{
					strcpy(s1,table[index1][index2].c_str());
				}
				strcat(s,s1);
				strcat(s,"//");
			}
			if(s[0]){
				strcat(s,pat.words[k].c_str());
				//x obtained, insert x
				map<char *, int, str_cmp>::iterator it;
				it=_xindex.find(s);
				if(it!=_xindex.end())
					feature_vector.push_back(it->second);
			}
			if(pat.end_of_group){//creat new clique
				clique cli;
				
				vector<node*> ns;
				for(k=0;k<pat.y.size();k++)
					ns.push_back(nod+i+pat.y[k]);
				node ** np=_test_clique_node.push_back(&ns[0],ns.size());
				cli.nodes=np;
				cli.node_num=ns.size();
				if(feature_vector.size()){
					cli.fvector=_test_clique_feature.push_back(&feature_vector[0],feature_vector.size());
				}else{
					cli.fvector=NULL;
				}
				cli.feature_num=feature_vector.size();
				cli.groupid=pat.groupid;
				cli.key=0;//random initialize
				clique *new_clique=_test_cliques.push_back(&cli,1);
				clisp.push_back(new_clique);
				feature_vector.clear();
			}
			
		}
		//set node -> clique
		if(clisp.size())
			nod[i].cliques = _test_node_clique.push_back(&clisp[0],clisp.size());
		else
			nod[i].cliques = NULL;
		nod[i].clique_num =clisp.size();
	}
	_test_nodes.free();
	_test_node_clique.free();
	_test_cliques.free();
	_test_clique_node.free();
	_test_clique_feature.free();
}


bool M3N::load_model(char *model_file){
/*
		_templets.clear();
		_templet_group.clear();
		_path2cliy.clear();
		_gsize=0;
		_order=0;
		_cols=0;
		_ysize=0;
		_tags.clear();
		_tag_str.clear();
		_xindex.clear();
		_x_freq.clear();
		_x_str.clear();
		_sequences.clear();
		_nodes.clear();
		_cliques.clear();
		_clique_node.clear();
		_node_clique.clear();
		_clique_feature.clear();
*/
	int i,j,k,ii;
	char line[MAXSTRLEN];
	ifstream fin;
	fin.open(model_file);
	if(!fin.is_open()){
		cout<<"model file: "<<model_file<<" not found"<<endl;
		return false;
	}
	//check version
	fin.getline(line,MAXSTRLEN-1);
	char *p=strstr(line,"\t");
	_version=atoi(p);
	printf("model version: 0.%d\n",_version);
	//load kernel type, kernel parameters
	fin.getline(line,MAXSTRLEN-1);
	_kernel_type=atoi(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_s=atof(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_d=atof(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_r=atof(line);

	//load eta
	fin.getline(line,MAXSTRLEN-1);
	_eta=atof(line);
	//load C
	fin.getline(line,MAXSTRLEN-1);
	_C=atof(line);
	//load freq_thresh
	fin.getline(line,MAXSTRLEN-1);
	_freq_thresh=atoi(line);

	//load templates
	int cur_group;
	while(!fin.eof()){
		fin.getline(line,MAXSTRLEN-1);
		if(!add_templet(line,cur_group))
			break;
	}
	_templets.back().end_of_group=true;
	_gsize=cur_group+1;
	_templet_group.resize(_gsize);
	cout<<"template number: "<<_templets.size()<<endl;
	//get ysize
	fin.getline(line,MAXSTRLEN-1);
	_ysize=atoi(line);
	_tags.resize(_ysize);
	for(i=0;i<_ysize;i++){
		fin.getline(line,MAXSTRLEN-1);
		char *q=_tag_str.push_back(line);
		_tags[i]=q;
	}
	cout<<"tags number: "<<_ysize<<endl;
	set_group();
	//get cols
	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	_cols=atoi(line);
	//load x
	int index;
	int x_num;
	
	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	x_num=atoi(line);
	for(i=0;i<x_num;i++){
		fin.getline(line,MAXSTRLEN-1);
		vector<char *> columns;
		split_string(line,"\t",columns);
		char *q=_x_str.push_back(columns[0]);
		index=atoi(columns[1]);
		_xindex.insert(make_pair(q,index));
	}

	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	if(_kernel_type==LINEAR_KERNEL){
		
		//load obj
		_obj=atof(line);
		fin.getline(line,MAXSTRLEN-1);
		//load _w
		_w_size=atoi(line);
		_w=new double[_w_size];
		for(i=0;i<_w_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_w[i]=atof(line);
		}
		fin.getline(line,MAXSTRLEN-1);
		//load _mu
		fin.getline(line,MAXSTRLEN-1);
		_mu_size=atoi(line);
		_mu=new double[_mu_size];
		for(i=0;i<_mu_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_mu[i]=atof(line);
		}
		cout<<_w_size<<" parameters loaded"<<endl;
		_get_kernel=&M3N::linear_kernel;
		_get_kernel_list=&M3N::linear_kernel;
	}else{
		//load _sequences info
		int seq_size=atoi(line);
		_sequences.resize(seq_size);
		for(i=0;i<_sequences.size();i++){
			sequence &seq=_sequences[i];
			
			fin.getline(line,MAXSTRLEN-1);
			sscanf(line,"%d\t%d",&seq.node_num,&seq.mu_index);
			node *nods=_nodes.alloc(seq.node_num);
			seq.nodes=nods;
			for(j=0;j<seq.node_num;j++){
				node &nod=nods[j];
				fin.getline(line,MAXSTRLEN-1);
				sscanf(line,"%d\t%d",&nod.key,&nod.clique_num);
				vector<clique*> clisp;
				for(k=0;k<nod.clique_num;k++){
					clique cli;
					fin.getline(line,MAXSTRLEN-1);
					sscanf(line,"%d\t%d\t%d\t%d",&cli.groupid,&cli.feature_num,&cli.node_num,&cli.key);
					vector<int> fvector(cli.feature_num);
					fin.getline(line,MAXSTRLEN-1);
					vector<char *> row;
					split_string(line,"\t",row);
					for(ii=0;ii<fvector.size();ii++){
						fvector[ii]=atoi(row[ii]);
					}
					if(fvector.size())
						cli.fvector=_clique_feature.push_back(&fvector[0],fvector.size());
					else
						cli.fvector=NULL;
					clique *new_clique=_cliques.push_back(&cli,1);
					clisp.push_back(new_clique);
				}
				if(clisp.size())
					nod.cliques=_node_clique.push_back(&clisp[0],clisp.size());
				else
					nod.cliques=NULL;
			}
		}
		//load _mu
		fin.getline(line,MAXSTRLEN-1);
		fin.getline(line,MAXSTRLEN-1);
		_obj=atof(line);
		fin.getline(line,MAXSTRLEN-1);
		_mu_size=atoi(line);
		_mu=new double[_mu_size];
		for(i=0;i<_mu_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_mu[i]=atof(line);
		}
		cout<<_mu_size<<" parameters loaded"<<endl;

		if(_kernel_type==POLY_KERNEL){
			_get_kernel=&M3N::poly_kernel;
			_get_kernel_list=&M3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&M3N::neural_kernel;
			_get_kernel_list=&M3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&M3N::rbf_kernel;
			_get_kernel_list=&M3N::rbf_kernel;
		}
		_clique_kernel.resize(_gsize);
		_path_kernel.resize(_path_num*_path_num);

	}
	fin.close();
	_path_num=pow((double)_ysize,_order+1);
	_node_anum=pow((double)_ysize,_order);//alpha(beta) number of each node
	_head_offset=-log((double)_ysize)*_order;
	switch(_kernel_type){
		case LINEAR_KERNEL: cout<<"linear kernel: k(a,b)=<a,b>"<<endl;break;
		case POLY_KERNEL: cout<<"polynomial kernel: k(a,b)=("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")^"<<_kernel_d<<endl;break;
		case RBF_KERNEL: cout<<"rbf kernel: k(a,b)=exp{-"<<_kernel_s<<"*||a-b||^2}"<<endl;break;
		case NEURAL_KERNEL: cout<<"neural kernel: k(a,b)=tanh("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")"<<endl;break;
	}
	return true;
}
