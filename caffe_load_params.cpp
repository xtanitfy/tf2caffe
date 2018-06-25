#include <caffe/caffe.hpp>  
#include <google/protobuf/io/coded_stream.h>  
#include <google/protobuf/io/zero_copy_stream_impl.h>  
#include <google/protobuf/text_format.h>  
#include <algorithm>  
#include <iosfwd>  
#include <memory>  
#include <string>  
#include <utility>  
#include <vector>  
#include <iostream>  
#include "caffe/common.hpp"  
#include "caffe/proto/caffe.pb.h"  
#include "caffe/util/io.hpp"  
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include <dirent.h>
#include <unistd.h>
#include <fstream>

using namespace caffe;  
using namespace std;  
using google::protobuf::io::FileInputStream;  
using google::protobuf::io::FileOutputStream;  
using google::protobuf::io::ZeroCopyInputStream;  
using google::protobuf::io::CodedInputStream;  
using google::protobuf::io::ZeroCopyOutputStream;  
using google::protobuf::io::CodedOutputStream;  
using google::protobuf::Message;  

#define NAME_MAX_LEN 64
#define TF_WEIGHTS_DIR "tf_model"

//#define ASSERT(x) ((x) || (printf("assertion failed ("__FILE__":%d): \"%s\"\n",__LINE__,#x), break_point(), false))
#define ASSERT(x) \
    if (!(x)) { \
        printf("[%s %s %d] error\n",__FILE__,__FUNCTION__,__LINE__); \
        exit(-1); \
    }
  
typedef float DATA_TYPE;

typedef enum
{
   TYPE_WEIHTS,
   TYPE_BATCHNORM_MEAN,
   TYPE_BATCHNORM_VARIANCE,
   TYPE_BATCHNORM_BETA,
   TYPE_BATCHNORM_GAMMA,
   TYPE_CNT
}PARAM_TYPE_e;

typedef struct 
{
    char netname[NAME_MAX_LEN];
    char layername[NAME_MAX_LEN];
    PARAM_TYPE_e type;
    int shape[4];
    int shape_len;
    DATA_TYPE *data;
    int len;
    char map_name[NAME_MAX_LEN*2];
    
}PARAM_INFO_s;

/*
struct {
    char *caffe_name;
    char *tf_name;
}map_layer_names [] = {
    {"conv1","Conv2d_0"},
    {"conv1","Conv2d_0"},
    {"conv1","Conv2d_0"},
    {"conv1","Conv2d_0"},
    {"conv1","Conv2d_0"},  
}
*/
map <string,vector <PARAM_INFO_s *> > maps;


const char *type_names[] = {"TYPE_WEIHTS","TYPE_BATCHNORM_MEAN",
"TYPE_BATCHNORM_VARIANCE","TYPE_BATCHNORM_BETA","TYPE_BATCHNORM_GAMMA"};

vector<string> split(string &str,const string &pattern);

int load_maps(vector <PARAM_INFO_s *> infos,char *filename)
{
    fstream f(filename);
    string      line; 
    while(getline(f,line)){
        vector <string > items;
        cout <<  "line:" << line.c_str() << endl;
        items = split(line," ");
        ASSERT(items.size() == 2 || items.size() == 3);
        
        vector <PARAM_INFO_s *> tf_names;
        for (int i = 1;i < items.size();i++) {
            cout <<  items[i].c_str() << " ";
            
            bool bfind = false;
            for (int j = 0;j < infos.size();j++) {
                if (strcmp(infos[j]->map_name,items[i].c_str()) == 0){
                    bfind = true;
                    tf_names.push_back(infos[j]);
                }
            }
            assert(bfind == true);
        }
        cout << endl;
        maps.insert(pair< string,vector <PARAM_INFO_s *> >(items[0], tf_names));
    }
    return 0;
}

int load_bin_data(char *dir,char *filename,PARAM_INFO_s *pInfo)
{
    char full_path[128];
    snprintf(full_path,sizeof(full_path),"%s/%s",dir,filename);
    
    FILE *fp = fopen(full_path,"rb");
    ASSERT(fp != NULL);
    
    cout << "full_path:" << full_path << endl;
    fseek(fp, 0L, SEEK_END);
    int file_size = ftell(fp);
    int size = 1;
    for (int j = 0;j < pInfo->shape_len;j++) {
        size *= pInfo->shape[j];
    }
    ASSERT(file_size/sizeof(DATA_TYPE) == size);
    
    pInfo->len = size;
    pInfo->data = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
    ASSERT(pInfo->data != NULL);
    
    fseek(fp, 0L, SEEK_SET);
    int nread = fread(pInfo->data,sizeof(DATA_TYPE),pInfo->len,fp);
    ASSERT(nread == pInfo->len);
    
    fclose(fp);
    
    return 0;
}


vector<string> split(string &str,const string &pattern)
{
    //const char* convert to char*
    char * strc = new char[strlen(str.c_str())+1];
    strcpy(strc, str.c_str());
    vector<string> resultVec;
    char* tmpStr = strtok(strc, pattern.c_str());
    while (tmpStr != NULL)
    {
        resultVec.push_back(string(tmpStr));
        tmpStr = strtok(NULL, pattern.c_str());
    }

    delete[] strc;

    return resultVec;
}

vector <PARAM_INFO_s *> parse_tf_weights(vector <string> &files)
{
    vector <PARAM_INFO_s *> infos;
    for (int i = 0;i < files.size();i++) {
        PARAM_INFO_s *pInfo = (PARAM_INFO_s *)malloc(sizeof(PARAM_INFO_s));
        ASSERT(pInfo != NULL);
        memset(pInfo,'\0',sizeof(PARAM_INFO_s));
        
        cout << "file:" << files[i].c_str() << endl;
        vector <string> file_infos = split(files[i],".");
        vector <string> items0 = split(file_infos[0],"=");
        ASSERT(items0.size() == 2);
        
        vector <string> shape_items = split(items0[1],"_");
        
        for (int j = 0;j < shape_items.size();j++) {
            pInfo->shape[pInfo->shape_len++] = atoi(shape_items[j].c_str());
        }
       
        vector <string> items1 = split(items0[0],"-");
        ASSERT(items1.size() == 3 || items1.size() == 4);
        
        snprintf(pInfo->netname,sizeof(pInfo->netname),"%s",items1[0].c_str());
        snprintf(pInfo->layername,sizeof(pInfo->netname),"%s",items1[1].c_str());
        
        strcpy(pInfo->map_name,items0[0].c_str()+strlen(pInfo->netname)+1);
        cout << "map_name:" << pInfo->map_name << endl;
        
        if (items1.size() == 3) {
            pInfo->type = TYPE_WEIHTS;  
        } else {
            ASSERT(strcmp(items1[2].c_str(),"BatchNorm") == 0);
            
            if (strcmp(items1[3].c_str(),"var") == 0) {
                pInfo->type = TYPE_BATCHNORM_VARIANCE;
                
            } else if (strcmp(items1[3].c_str(),"mean") == 0) {
                pInfo->type = TYPE_BATCHNORM_MEAN;
                
            } else if (strcmp(items1[3].c_str(),"gamma") == 0) {
                pInfo->type = TYPE_BATCHNORM_GAMMA;
                
            } else if (strcmp(items1[3].c_str(),"beta") == 0) {
                pInfo->type = TYPE_BATCHNORM_BETA;
                
            }
        }
        
        load_bin_data((char *)TF_WEIGHTS_DIR,(char *)files[i].c_str(),pInfo);   
        infos.push_back(pInfo);
    }
    
    return infos;
}

/*
void map_name(char *tf_name,char *map_name)
{
    if (strcmp(tf_name,"Logits") == 0) {
        strcpy(map_name,"fc7"); 
    }  else if ()
}
*/
vector <PARAM_INFO_s *> find_info(LayerParameter *p_layer_param)
{
    char *prototxt_layername = (char *)p_layer_param->name().c_str();
    map <string,vector <PARAM_INFO_s *> >::iterator iter;
    bool bfind = false;
    vector <PARAM_INFO_s *>info;
    for(iter = maps.begin(); iter != maps.end(); iter++){
        string layername = iter->first;
        if (strcmp(prototxt_layername,layername.c_str()) == 0) {
            info = iter->second;
            bfind = true;
        }
    }
    ASSERT(bfind == true);
    
    return info;
    
}

int save_one_blob(PARAM_INFO_s *info,LayerParameter *p_layer_param)
{
    vector<int> blob_shape;
    for (int i = 0;i < info->shape_len;i++) {
        blob_shape.push_back(info->shape[i]);
    }
    Blob<DATA_TYPE> blob(blob_shape);
    
    DATA_TYPE *pData = blob.mutable_cpu_data();
    for (int i = 0; i < blob.count(); ++i) {
        pData[i] = info->data[i];
    }
    blob.ToProto(p_layer_param->add_blobs(),false);
}

//ok then finish it! As bathnorm_gamma caffe in ScaleParameter
int load_conv(LayerParameter *p_layer_param)
{
    vector <PARAM_INFO_s *> info = find_info(p_layer_param);
    ASSERT(info.size() == 1);
    save_one_blob(info[0],p_layer_param);

    return 0;
}

int load_batchnorm(LayerParameter *p_layer_param)
{
    vector <PARAM_INFO_s *> info = find_info(p_layer_param);
    ASSERT(info.size() == 2);
    
    for (int i = 0;i < info.size();i++) {
        save_one_blob(info[i],p_layer_param);
    }
    vector<int> blob_shape;
    blob_shape.push_back(1);
    
    Blob<DATA_TYPE> blob(blob_shape);
    DATA_TYPE *pData = blob.mutable_cpu_data();
    pData[0] = 1.0;
    
    blob.ToProto(p_layer_param->add_blobs(),false);
    
    return 0;
}

int load_bias(LayerParameter *p_layer_param)
{
    vector <PARAM_INFO_s *> info = find_info(p_layer_param);
    ASSERT(info.size() == 1);
    
    for (int i = 0;i < info.size();i++) {
        save_one_blob(info[i],p_layer_param);
    }

    return 0;
}

int load_scale(LayerParameter *p_layer_param)
{
    vector <PARAM_INFO_s *> info = find_info(p_layer_param);
    ASSERT(info.size() == 1);
    
    for (int i = 0;i < info.size();i++) {
        save_one_blob(info[i],p_layer_param);
    }

    return 0;
}

int load_scale_ext(LayerParameter *p_layer_param)
{
    vector <PARAM_INFO_s *> info = find_info(p_layer_param);
    ASSERT(info.size() == 2);
    
    for (int i = 0;i < info.size();i++) {
        save_one_blob(info[i],p_layer_param);
    }

    return 0;
}


vector <string> get_all_filenames(char *basepath)
{
    vector <string> files;
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
 
    if ((dir = opendir(basepath)) == NULL) {
        perror("Open dir error...");
        exit(1);
    }
    while ((ptr=readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8)  {
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            if (strstr(ptr->d_name,".bin") != NULL) {
                files.push_back(ptr->d_name);
            }
        }
        else if(ptr->d_type == 10)    {
            //link file 

        }
        else if(ptr->d_type == 4)  {
            // dir 
        }
    }
    closedir(dir);
    return files;
}


int main()  
{  
    NetParameter proto;  
    ReadProtoFromTextFile("caffe_model/deploy.prototxt", &proto);
    vector <string> files;
    /*
    for (int i = 0;i < proto.layer_size();i++) {
        LayerParameter *p_layer_param = (LayerParameter *)&proto.layer(i);
        printf("%s \n",p_layer_param->name().c_str());
        char *type = (char *)p_layer_param->type().c_str();
        printf("type:%s\n",type);
    }
     printf("proto.layer_size():%d\n",proto.layer_size());
     getchar();
    */
    files = get_all_filenames((char *)TF_WEIGHTS_DIR);
    for (int i = 0;i < files.size();i++) {
        cout << files[i].c_str() << endl;
    }
    //getchar();
    
    vector <PARAM_INFO_s *> infos = parse_tf_weights(files);
    load_maps(infos,(char *)"layer_map.txt");
    map <string,vector <PARAM_INFO_s *> >::iterator iter;
    cout << "---------------print maps:" << endl;
    for(iter = maps.begin(); iter != maps.end(); iter++){
        
        cout<<iter->first<<"->";
        vector <PARAM_INFO_s *>info = iter->second;
        for (int i = 0;i < info.size();i++) {
            cout << info[i]->map_name << " ";
        }
        cout << endl;
    }
    
       
    
    for (int i = 0;i < infos.size();i++) {
        
        cout << "layername:" << infos[i]->layername << endl;
        cout << "type:" << type_names[infos[i]->type] << endl;
        cout << "shape: ";
        for (int j = 0;j < infos[i]->shape_len;j++) {
            cout << infos[i]->shape[j] << " ";
        }
        cout << endl;
    }
    getchar();
   
    printf("proto.layer_size():%d\n",proto.layer_size());
    for (int i = 0;i < proto.layer_size();i++) {
        LayerParameter *p_layer_param = (LayerParameter *)&proto.layer(i);
        printf("%s \n",p_layer_param->name().c_str());
        //printf("    type:%s \n",p_layer_param->type().c_str());
        char *type = (char *)p_layer_param->type().c_str();
        if (strcmp(type,"Convolution") == 0 || strcmp(type,"ConvolutionDepthwise") == 0) {
            load_conv(p_layer_param);
            
        } else if (strcmp(type,"BatchNorm") == 0) {
            load_batchnorm(p_layer_param);
            
        } else if (strcmp(type,"Scale") == 0) {
            load_scale_ext(p_layer_param);
        } else if (strcmp(type,"Bias") == 0) {
            
            load_bias(p_layer_param);
        }
        //printf("type:%s\n",type);
        //getchar();
        
        printf("    blobs_size:%d \n",p_layer_param->blobs_size());
        for (int j = 0;j < p_layer_param->blobs_size();j ++) {
            BlobShape blobshape = p_layer_param->blobs(j).shape();
            cout << "      ";
            for (int k = 0;k < blobshape.dim_size();k++) {
                cout << blobshape.dim(k) << " ";
            }
            cout << endl;
        }
    }
    
    
    WriteProtoToBinaryFile(proto, "caffe_model/test.caffemodel");  
    return 0;  
} 


