#define PCL_NO_PRECOMPILE
#include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include<cmath>


const float voxel_resolution = 0.008f;
const float seed_resolution = 0.1f;
const float spatial_importance = 1.0f;
const float color_importance = 0.0f;
const float normal_importance = 0.0f;

struct EIGEN_ALIGN16 GaussianPoint
{
    PCL_ADD_POINT4D;
    PCL_ADD_RGB;
    float normal_x;  float normal_y;  float normal_z;
    float rot_0;  float rot_1;  float rot_2;  float rot_3;
    uint32_t lable;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 确保对齐内存
};

//(type, data_structure_attr, ply_attr)
POINT_CLOUD_REGISTER_POINT_STRUCT(GaussianPoint, 
        (float, x, x)(float, y, y)(float, z, z)
        (uint8_t, r, f_dc_0)(uint8_t, g, f_dc_1)(uint8_t, b, f_dc_2)
        (uint8_t, a, opacity)
        (float, normal_x, scale_0)(float, normal_y, scale_1)(float, normal_z, scale_2)
        (float, rot_0, rot_0)(float, rot_1, rot_1)(float, rot_2, rot_2)(float, rot_3, rot_3)
        (uint32_t, lable, lable)
)

// Types
typedef GaussianPoint PointT;
// typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;


/*
https://pcl.readthedocs.io/projects/tutorials/en/master/supervoxel_clustering.html#supervoxel-clustering
*/
int main(int argc, char ** argv){
    if (argc < 2){
        pcl::console::print_error ("Syntax is: %s <pcd-file> \n "
                                    "--NT Dsables the single cloud transform \n"
                                    "-v <voxel resolution>\n-s <seed resolution>\n"
                                    "-c <color weight> \n-z <spatial weight> \n"
                                    "-n <normal_weight>\n", argv[0]);
        return (1);
    }

    PointCloudT::Ptr cloud (new PointCloudT);
    if (pcl::io::loadPLYFile<PointT> (argv[1], *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file");
    }
    printf("There are %lu points\n", cloud->points.size());
    printf("%f %f %f\n",cloud->points[0].x, cloud->points[0].y, cloud->points[0].z);
    // printf("%u %u %u\n",cloud->points[0].r, cloud->points[0].g, cloud->points[0].b);
    // printf("%f %f %f\n",cloud->points[0].normal_x, cloud->points[0].normal_y, cloud->points[0].normal_z);
    std::cout<< sizeof( cloud->points[0]) <<std::endl;
    pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);


    super.setInputCloud (cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);

    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);
    int num_clusters = supervoxel_clusters.size();
    printf("There are %d supervoxels\n", num_clusters);

    PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();

    // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    // viewer->setBackgroundColor (0, 0, 0);
    // viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
    // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");
    // getchar();


    pcl::io::savePLYFile ("../test2.ply", *labeled_voxel_cloud);
    int total = 0;
    for(auto&x: supervoxel_clusters){
        total += x.second->voxels_->points.size();
        // printf("%ld\n",x.second->voxels_->points.size());
    }
    printf("%d\n", total);
    
    return num_clusters;
}