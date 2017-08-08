#include <iostream>
#include <string>

#include <gperftools/profiler.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/console/time.h>
#include <Eigen/Dense>  

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void
convertToColor (PointCloudT::ConstPtr input,std::vector<int> &indices,pcl::PointCloud<pcl::PointXYZRGBA> &output,int r,int g,int b,int a)
{
    for (std::vector<int>::iterator pit = indices.begin(); pit!= indices.end(); pit++ )
    {
      pcl::PointXYZRGBA n_point;
      n_point.x = input->points[*pit].x;
      n_point.y = input->points[*pit].y;
      n_point.z = input->points[*pit].z;
      n_point.r = r;
      n_point.g = g;
      n_point.b = b;
      n_point.a = a;
      output.points.push_back(n_point);
    }
}
void 
sampleFunction (PointCloudT::ConstPtr input,PointCloudT &output)
{
  pcl::console::TicToc time;
  time.tic ();
  pcl::RandomSample<PointT> sam;
  sam.setSample(20000);
  sam.setInputCloud(input);
  sam.filter(output);
  std::cout<<"sample---"<<time.toc()<<"-------"<<input->points.size()<<"---to---"<<output.points.size()<<std::endl;
}

void 
clusterFunction (PointCloudT::ConstPtr input,pcl::PointCloud<pcl::PointXYZRGBA> &segmentcloud)
{
  pcl::console::TicToc time;
  time.tic ();
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(input);
  pcl::EuclideanClusterExtraction<PointT> ec;
  std::vector<pcl::PointIndices> cluster_indices;
  ec.setClusterTolerance(0.6);
  ec.setMinClusterSize(60);
  ec.setMaxClusterSize(25000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(input);
  ec.extract(cluster_indices);
  std::cout<<"ec-complete------"<<cluster_indices.size()<<"------"<<time.toc()<<std::endl;
  
  pcl::console::TicToc time_merge;
  time_merge.tic ();

  std::vector<PointCloudT::Ptr> vec_cloudptrs;
  for(std::vector<pcl::PointIndices>::iterator it = cluster_indices.begin();it!= cluster_indices.end();it++ )
  {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
    int a=255,r=rand()%255,g=rand()%255,b=rand()%255;
    convertToColor(input,it->indices,*cloud_cluster_ptr,r,g,b,a);
    segmentcloud+=*cloud_cluster_ptr;
  }
  std::cout<<"merge-complete------"<<time_merge.toc()<<std::endl;
}

void 
sacFunction (PointCloudT::ConstPtr input,PointCloudT &output,pcl::PointCloud<pcl::PointXYZRGBA> &groundcloud)
{
  pcl::PointIndices::Ptr inlier(new pcl::PointIndices);
  pcl::console::TicToc time;
  time.tic ();
  //std::cout<<"load"<<std::endl;
  pcl::ExtractIndices<PointT> filter(false);
  pcl::ModelCoefficients::Ptr cof(new pcl::ModelCoefficients);
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setModelType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.2);
  seg.setInputCloud(input);
  seg.segment(*inlier,*cof);
  //std::cout<<seg.getProbability()<<std::endl;
  convertToColor(input,inlier->indices,groundcloud,0,0,255,255);
  //pcl::copyPointCloud(*input,*inlier,groundcloud);
  filter.setInputCloud(input);
  filter.setIndices(inlier);
  filter.setNegative(true);
  filter.filter(output);
  std::cout<<"ground_segment-complete---"<<time.toc()<<std::endl;
}

void 
getSegmentCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmentcloud,PointCloudT::Ptr cloud)
{   
    
    //pcl::PointCloud<pcl::PointXYZI>::Ptr rawcloud(new pcl::PointCloud<pcl::PointXYZI>);
    //PointCloudT::Ptr cloud(new PointCloudT);
    PointCloudT::Ptr outputcloud(new PointCloudT);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr groundcloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    
    pcl::console::TicToc time;
    time.tic();
    segmentcloud->points.clear();
    //pcl::console::TicToc timeload;
    //timeload.tic ();
    //pcl::PCLPointCloud2 cloud_blob;
    //pcl::io::loadPCDFile ("./pcd/19.pcd", cloud_blob);
    //pcl::fromPCLPointCloud2 (cloud_blob, *rawcloud); 
    //pcl::io::loadPCDFile ("./pcd/"+  std::to_string(n)+".pcd", *rawcloud);
    //pcl::copyPointCloud(*rawcloud,*cloud);
    //std::cout<<"load---"<<rawcloud->points.size()<<"------"<<timeload.toc ()<<std::endl;
    
    sampleFunction(cloud,*outputcloud);
    *cloud = *outputcloud;

    sacFunction(cloud,*outputcloud,*groundcloud);
    *cloud = *outputcloud;

    clusterFunction(cloud,*segmentcloud);

    *segmentcloud += *groundcloud;

    std::cout<<"总时间---"<<time.toc ()<<std::endl;
    std::cout<<"--------------------------"<<std::endl;
}
void
LoadPCD(boost::function<void( PointCloudT::ConstPtr,int)> function)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr rawcloud(new pcl::PointCloud<pcl::PointXYZI>);
  PointCloudT::Ptr cloud(new PointCloudT);
  boost::signals2::signal<void( PointCloudT::ConstPtr,int)> sig;
  sig.connect(function);
  int n = 0;
  while(n <= 76)
  {
    pcl::io::loadPCDFile ("./pcd/"+  std::to_string(n)+".pcd", *rawcloud);
    std::cout<<"output---"<<n<<std::endl;
    pcl::copyPointCloud(*rawcloud,*cloud);
    sig(cloud,1);
    boost::this_thread::sleep(boost::posix_time::microseconds (50000));
    n++;
  }
  sig(cloud,0);
  
}
void
Secondthread(boost::function<void( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr )> function)
{
  PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cloud_back(new PointCloudT);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmentcloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  boost::signals2::signal<void( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr)> sig;
  sig.connect(function);
  int tag_cloud = 1;
  int size_cloud = 1;
  boost::mutex mutex;
  boost::function<void( PointCloudT::ConstPtr,int)> function_load =
      [ cloud,&mutex,&tag_cloud]( PointCloudT::ConstPtr ptr,int tag )
      {
        boost::mutex::scoped_lock lock( mutex );
        *cloud = *ptr;
        tag_cloud = tag;
      };
  boost::thread th_load(&LoadPCD,function_load);
  boost::this_thread::sleep(boost::posix_time::microseconds (100000));
  while(true)
  {
    if(true)
    {
        boost::mutex::scoped_try_lock lock( mutex );
        if( lock.owns_lock())
        {
          if (tag_cloud == 0)
            break;
          *cloud_back = *cloud;
       }
    }
    if(size_cloud == cloud_back->points.size())
    {
      std::cout<<"skip---lock:10000"<<std::endl;
      boost::this_thread::sleep(boost::posix_time::microseconds (10000));
      continue;
    } 
    else
    {
      size_cloud = cloud_back->points.size();
      getSegmentCloud(segmentcloud,cloud_back);
      sig(segmentcloud);

    }
  }
      
  
}
int
main (int argc,
      char* argv[])
{ 
  ProfilerStart("./profile.prof");

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmentcloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::visualization::PCLVisualizer viewer("Segment");
  viewer.setBackgroundColor(0.0,0.0,0.0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> handler_s(segmentcloud);
  boost::mutex mutex;
  boost::function<void( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr )> function =
      [ segmentcloud,&mutex,&viewer,&handler_s]( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr ptr )
      {
        boost::mutex::scoped_lock lock( mutex );
        *segmentcloud = *ptr;
        if(!viewer.updatePointCloud(segmentcloud,handler_s,"segment"))
        {
          viewer.addPointCloud(segmentcloud,handler_s,"segment");
        }
        viewer.spinOnce ();

      };
  boost::thread th(&Secondthread,function);
  

  while (!viewer.wasStopped ())
  {
    //boost::this_thread::sleep(boost::posix_time::microseconds (100000));
  }
  ProfilerStop();
  return (0);
}