#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include "m3n.h"
using namespace std;
void test(M3N *m, char *model_file, char *key_file, char *result_file, int margin, int nbest){
	vector < vector < string > > table;
	int total=0;
	int error=0;
	ifstream fin;
	fin.open(key_file);
	ofstream fout;
	fout.open(result_file);
	char line[10000];
	int i,j;
	while(!fin.eof()){
		fin.getline(line,9999);
		if(line[0]){
			vector < string > row;
			char *p=line,*q;
			while(q=strstr(p,"\t"))
			{
				*q=0;
				row.push_back(p);
				p=q+1;
			}
			row.push_back(p);
			table.push_back(row);
		}else if(table.size()){
			vector < vector < string > > y;
			vector < double > sequencep;
			vector < vector < double > > nodep;
			m->tag(table,y,sequencep,nodep);
			if(margin){
				for(i=0;i < sequencep.size()-1 ; i++)
					fout<<sequencep[i]<<"\t";
				fout<<sequencep[i]<<endl;
			}
			int table_rows=y[0].size();
			int table_cols=table[0].size();
			total+=table_rows;
			for(i=0;i < table_rows ; i++)
			{
				if(table[i].back()!=y[0][i])
					error++;
				for(j=0;j < table_cols;j++)
					fout<<table[i][j].c_str()<<"\t";
				for(j=0;j < y.size()-1;j++)
					fout<<y[j][i].c_str()<<"\t";
				fout<<y[j][i].c_str();
				if(margin){
					fout<<"\t";
					for(j=0;j < nodep[0].size()-1; j++)
						fout<<nodep[i][j]<<"\t";
					fout<<nodep[i][j];
				}
				fout<<endl;
			}
			table.clear();
			fout<<endl;
		}
	}
	if(table.size()){
		vector < vector < string > > y;
		vector < double > sequencep;
		vector < vector < double > > nodep;
		m->tag(table,y,sequencep,nodep);
		if(margin){
			for(i=0;i < sequencep.size()-1 ; i++)
				fout<<sequencep[i]<<"\t";
			fout<<sequencep[i]<<endl;
		}
		int table_rows=y[0].size();
		int table_cols=table[0].size();
		total+=table_rows;
		for(i=0;i < table_rows ; i++)
		{
			if(table[i].back()!=y[0][i])
				error++;
			for(j=0;j < table_cols;j++)
				fout<<table[i][j].c_str()<<"\t";
			for(j=0;j < y.size()-1;j++)
				fout<<y[j][i].c_str()<<"\t";
			fout<<y[j][i].c_str();
			if(margin){
				fout<<"\t";
				for(j=0;j < nodep[0].size()-1; j++)
					fout<<nodep[i][j]<<"\t";
				fout<<nodep[i][j];
			}
			fout<<endl;
		}
		table.clear();
		fout<<endl;
	}
	fin.close();
	fout.close();
	cout<<"label precision:"<<(1-(double)error/total)<<endl;
}

//build learn
int main_learn(int argc, char *argv[]){
const char* learn_help="\
usage:\n\
m3n learn template_file_name train_file_name model_file_name\n\
option type   default   meaning\n\
-c     double 1         Slack variables penalty factor\n\
-f     int    0         Frequency threshold.\n\
-k     int    0         Kernel type. \n\
                        0: linear kernel <a,b>\n\
                        1: polynomial kernel (s*<a,b>+r)^d\n\
                        2: rbf kernel exp{-s*||a-b||^2}\n\
                        3: neural kernel tanh(s*<a,b>+r)\n\
-s     double 1         \n\
-d     int    1         \n\
-r     double 0         \n\
-i     int    10        Max iteration number. \n\
-e     double 0.000001  Controls training precision \n\
-o     int    0         With higher value, more information is printed.\
";
//initial learning parameters
	char train_file[100]="";
	char templet_file[100]="";
	char model_file[100]="";
	bool relearn=false;			//-x
	char C[100]="1";			//-c
	char freq_thresh[100]="0";	//-f
	char kernel_type[100]="0";	//-k
	char kernel_s[100]="1";		//-s
	char kernel_d[100]="1";		//-d
	char kernel_r[100]="0";		//-r
	char max_iter[100]="10";	//-i
	char eta[100]="0.000001";	//-e
	char print_level[100]="0";	//-o
	//get learning parameters, and check
	int i=2;
	int step=0;//0: next load templet_file, 1: next load train_file, 2: next_load model_file
	while(i<argc){
		if(!strcmp(argv[i],"-x")){
			relearn=true;
			i++;
		}else if(!strcmp(argv[i],"-h")){
			cout<<learn_help<<endl;
			return 0;
		}else if(!strcmp(argv[i],"-c")){
			if(i+1==argc){
				cout<<"-c parameter empty"<<endl;
				return 1;
			}
			strcpy(C,argv[i+1]);
			if(atof(C)<=0){
				cout<<"invalid -c parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-f")){
			if(i+1==argc){
				cout<<"-f parameter empty"<<endl;
				return 1;
			}
			strcpy(freq_thresh,argv[i+1]);
			if(atoi(freq_thresh)<0){
				cout<<"invalid -f parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-k")){
			if(i+1==argc){
				cout<<"-k parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_type,argv[i+1]);
			if(atoi(kernel_type)<0||atoi(kernel_type)>3){
				cout<<"invalid -k parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-s")){
			if(i+1==argc){
				cout<<"-s parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_s,argv[i+1]);
			if(atof(kernel_s)<0){
				cout<<"invalid -s parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-d")){
			if(i+1==argc){
				cout<<"-d parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_d,argv[i+1]);
			if(atoi(kernel_d)<1){
				cout<<"invalid -d parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-r")){
			if(i+1==argc){
				cout<<"-r parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_r,argv[i+1]);
			if(atof(kernel_r)<0){
				cout<<"invalid -r parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-i")){
			if(i+1==argc){
				cout<<"-i parameter empty"<<endl;
				return 1;
			}
			strcpy(max_iter,argv[i+1]);
			if(atoi(max_iter)<0){
				cout<<"invalid -i parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-e")){
			if(i+1==argc){
				cout<<"-e parameter empty"<<endl;
				return 1;
			}
			strcpy(eta,argv[i+1]);
			if(atof(eta)<0){
				cout<<"invalid -e parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-o")){
			if(i+1==argc){
				cout<<"-o parameter empty"<<endl;
				return 1;
			}
			strcpy(print_level,argv[i+1]);
			if(atoi(print_level)<0){
				cout<<"invalid -o parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(argv[i][0]=='-'){
			cout<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}else if(step==0){
			strcpy(templet_file,argv[i]);
			i++;
			step++;
		}else if(step==1){
			strcpy(train_file,argv[i]);
			i++;
			step++;
		}else if(step==2){
			strcpy(model_file,argv[i]);
			i++;
			step++;
		}
	}
	//check necessary parameters
	if(!templet_file[0]){
		cout<<"no template file"<<endl;
		return 1;
	}
	if(!train_file[0]){
		cout<<"no train file"<<endl;
		return 1;
	}
	if(!model_file[0]){
		cout<<"no model file"<<endl;
		return 1;
	}
	M3N *m=new M3N();
	m->set_para("C",C);
	m->set_para("freq_thresh",freq_thresh);
	m->set_para("kernel_type",kernel_type);
	m->set_para("kernel_s",kernel_s);
	m->set_para("kernel_d",kernel_d);
	m->set_para("kernel_r",kernel_r);
	m->set_para("max_iter",max_iter);
	m->set_para("eta",eta);
	m->set_para("print_level",print_level);
	m->learn(templet_file,train_file,model_file,relearn);
	delete m;
	return 0;
}
int main_test(int argc, char *argv[]){
const char* test_help="\
usage:\n\
m3n test model_file_name key_file_name result_file_name\n\
option type   default   meaning\n\
-m     int    0         Whether output marginal probability.\n\
-n     int    1         Output n best results.\n\
";
//initial testing parameters
	char model_file[100]="";
	char key_file[100]="";
	char result_file[100]="";
	char margin[100]="0";		//-m
	char nbest[100]="1";		//-n
	


	int i=2;
	int step=0;//0: next load model_file, 1: next load key_file,2: result_file
	while(i<argc){
		if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cout<<"-m parameter empty"<<endl;
				return 1;
			}
			strcpy(margin,argv[i+1]);
			if(atoi(margin)!=0 && atoi(margin)!=1){
				cout<<"invalid -m parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-n")){
			if(i+1==argc){
				cout<<"-n parameter empty"<<endl;
				return 1;
			}
			strcpy(nbest,argv[i+1]);
			if(atoi(nbest)<1){
				cout<<"invalid -n parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-h")){
			cout<<test_help<<endl;
			return 0;
		}else if(argv[i][0]=='-'){
			cout<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}else if(step==0){
			strcpy(model_file,argv[i]);
			i++;
			step++;
		}else if(step==1){
			strcpy(key_file,argv[i]);
			i++;
			step++;
		}else if(step==2){
			strcpy(result_file,argv[i]);
			i++;
			step++;
		}
	}
	//check necessary parameters
	if(!model_file[0]){
		cout<<"no model file"<<endl;
		return 1;
	}
	if(!key_file[0]){
		cout<<"no key file"<<endl;
		return 1;
	}if(!result_file[0]){
		cout<<"no result file"<<endl;
		return 1;
	}


	M3N *m=new M3N();
	m->set_para("margin",margin);
	m->set_para("nbest",nbest);
	m->load_model(model_file);
	test(m,model_file,key_file,result_file,atoi(margin),atoi(nbest));
	delete m;
	return 0;
}


int main(int argc, char *argv[]){
	if(argc < 2 || strcmp(argv[1],"learn") && strcmp(argv[1],"test")){
		printf("usage:\n");
		printf("m3n learn template_file_name train_file_name model_file_name\n");
		printf("m3n test template_file_name train_file_name model_file_name\n");
		return 0;
	}
	if(!strcmp(argv[1],"learn"))
		return main_learn(argc,argv);
	else
		return main_test(argc,argv);
}
