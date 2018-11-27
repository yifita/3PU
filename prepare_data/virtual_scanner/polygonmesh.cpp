#include <string>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>
#include <stdexcept>

#include "CMesh.h"

#include "ParameterMgr.h"
#include "GlobalFunction.h"
#include "Camera.h"
#include "DataMgr.h"

#include "vcg/complex/algorithms/update/position.h"
#include "vcg/complex/algorithms/update/normal.h"
#include "vcg/complex/algorithms/update/bounding.h"
#include "vcg/complex/algorithms/point_sampling.h"

//#include "vcg/complex/algorithms/intersection.h"


using std::vector;

using namespace vcg;
using namespace std;

#include <iostream>

//typedef tri::MeshSampler<CMesh> BaseSampler;


class BaseSampler
{
    public:
        BaseSampler(CMesh* _m){ m = _m; uvSpaceFlag = false; qualitySampling = false; tex = 0; };
        CMesh *m;
        QImage* tex;
        int texSamplingWidth;
        int texSamplingHeight;
        bool uvSpaceFlag;
        bool qualitySampling;

        void reset()
        {
          m->Clear();
        }

        void AddVert(const CMesh::VertexType &p)
        {
            tri::Allocator<CMesh>::AddVertices(*m, 1);
            m->vert.back().ImportData(p);
        }

        void AddFace(const CMesh::FaceType &f, CMesh::CoordType p)
        {
          tri::Allocator<CMesh>::AddVertices(*m,1);
          m->vert.back().P() = f.cP(0)*p[0] + f.cP(1)*p[1] +f.cP(2)*p[2];
        }

    void samplePointsFromMesh(CMesh& mesh, CMesh* points)
    {
        mesh.bbox.SetNull();
        for (int i = 0; i < mesh.vert.size(); i++)
        {
            mesh.bbox.Add(mesh.vert[i]);
        }
        mesh.vn = mesh.vert.size();
        mesh.fn = mesh.face.size();
        vcg::tri::UpdateNormal<CMesh>::PerVertex(mesh);

        float radius = 0;
        //int sampleNum = para->getDouble("Poisson Disk Sample Number");
        int sampleNum = 3000;
      //int sampleNum = global_paraMgr.registration.getInt("Poisson Samling Number");
      if (sampleNum <= 100)
        {
            sampleNum = 100;
        }
        radius = tri::SurfaceSampling<CMesh, BaseSampler>::ComputePoissonDiskRadius(mesh, sampleNum);
        // first of all generate montecarlo samples for fast lookup
        CMesh *presampledMesh = &(mesh);
        CMesh MontecarloMesh; // this mesh is used only if we need real poisson sampling (and therefore we need to choose points different from the starting mesh vertices)


        BaseSampler sampler(&MontecarloMesh);
        sampler.qualitySampling = true;
        tri::SurfaceSampling<CMesh, BaseSampler>::Montecarlo(mesh, sampler, sampleNum * 20);
        MontecarloMesh.bbox = mesh.bbox; // we want the same bounding box
        presampledMesh = &MontecarloMesh;


        BaseSampler mps(points);
        tri::SurfaceSampling<CMesh, BaseSampler>::PoissonDiskParam pp;
        //tri::SurfaceSampling<CMesh, BaseSampler>::PoissonDisk(mesh, mps, *presampledMesh, radius, pp);

        //cout << "before pruning " << presampledMesh->vert.size() << endl;

        tri::SurfaceSampling<CMesh,BaseSampler>::PoissonDiskPruning(mps, *presampledMesh, radius, pp);
    }


    void updateMesh(CMesh* mesh)
    {
        tri::UpdatePosition<CMesh>::Matrix(*mesh, mesh->Tr);
        tri::UpdateNormal<CMesh>::PerVertexMatrix(*mesh,mesh->Tr);
        tri::UpdateNormal<CMesh>::PerFaceMatrix(*mesh,mesh->Tr);
        tri::UpdateBounding<CMesh>::Box(*mesh);
        mesh->Tr.SetIdentity();
    }
};

int	main(int argc, char *argv[]) {
    double starting_resolution = 56;
    int num_resolutions = 6;
    srand (time(NULL));

    DataMgr dataMgr(global_paraMgr.getDataParameterSet());

    CMesh views;
    int mask = tri::io::Mask::IOM_ALL;
    tri::io::Importer<CMesh>::Open(views, "sphere_view_points_1k_perspective_700_wlop_ordered_cut2.ply", mask);

    QString input_model_base("mesh_data");
    string output_scan_base = "output";
    if (argc > 1)
    {
        input_model_base = argv[1];
        output_scan_base = argv[2];
    }


    cout << "model path " << input_model_base.toStdString() << endl;
    QDir dir(input_model_base);
    if (!dir.exists())
    {
        cout << "model path wrong" << endl;
        return 1;
    }

    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    int model_num = list.size();
    cout << "input_model_base number: " << list.size() << endl;

    Matrix44f trRot; trRot.SetIdentity();

    Point3f axis(0, 0, 1);

    int start_id = 0;
    int end_id = 50;
    int model_id = int(rand() % model_num);
    if (argc > 3)
    {
        start_id = atoi(argv[3]);
        end_id = atoi(argv[4]);
        model_id = atoi(argv[5]);
    }

    QFileInfo fileInfo = list.at(model_id);
    QString f_name = fileInfo.fileName();
    QString input_model_path = input_model_base + "/" + f_name;
    cout << "pick_model: " << input_model_path.toStdString() << endl;
    dataMgr.loadPlyToOriginal(input_model_path);

    CMesh* occludee = dataMgr.getCurrentOriginal();

    for(int scan_id = start_id; scan_id < end_id; scan_id++)
    {
        // virtual scan
        int cam_id = int(rand() % views.vert.size());
        cout << "cam id: " << cam_id << endl;
        Point3f cam_pos = views.vert[cam_id];
        //cout << "camera position " << cam_pos.X() << " " << cam_pos.Y() << " " << cam_pos.Z() << " "<< endl;

        vcc::Camera camera;
        camera.setParameterSet(global_paraMgr.getCameraParameterSet());
        camera.setInput(&dataMgr);
        camera.setPosition(views.vert[cam_id]);

        for (int res_id = 0; res_id < num_resolutions; ++res_id)
        {
            float curr_resolution = starting_resolution * sqrt(double(1<<res_id));
            global_paraMgr.camera.setValue("Camera Resolution", DoubleValue(1/curr_resolution));
            Timer time;
            time.start("Scan!!");

            global_paraMgr.camera.setValue("Run Virtual Scan", BoolValue(true));

            camera.run();
            camera.clear();
            global_paraMgr.camera.setValue("Run Virtual Scan", BoolValue(false));
            CMesh* scan_data = dataMgr.getCurrentScannedMesh();

            cout << "scanned number: " << scan_data->vert.size() << endl;

            time.end();

    //        int mask = tri::io::Mask::IOM_ALL + tri::io::Mask::IOM_VERTNORMAL;
            int mask= tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL;
            mask += tri::io::Mask::IOM_VERTCOLOR;

            string f_namestr = f_name.toStdString();
            string output_dir = output_scan_base + to_string(int(curr_resolution));
            string output_scan_path = output_dir + "/" + f_namestr.substr(0, f_namestr.find_last_of(".")) + "_" + to_string(scan_id) + ".ply";
            const int dir_err = mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (dir_err == -1)
            {
                if( errno == EEXIST ) {}
                else {
                   // something else
                    std::cout << "cannot create " << output_dir << " error:" << strerror(errno) << std::endl;
                    throw std::runtime_error( strerror(errno) );
                }
            }
            cout << "Saving to " << output_scan_path << endl;
            tri::io::ExporterPLY<CMesh>::Save(*scan_data, output_scan_path.c_str(), mask, false);
        }
    }

    return 0;
}
