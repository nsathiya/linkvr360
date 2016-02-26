{
  "targets": [
    {
      "target_name": "addon",
      "sources": [ "hello.cc", "test.cpp", "ImageOps.cpp", "CameraOps.cpp", "GPUOps.cpp", "MemoryManager.cpp", "BlenderOps.cpp" ],
      'cflags' : ['-Wno-unused-variable'],
		'cflags!': [ '-fno-exceptions'],
		'cflags_cc!': [ '-fno-exceptions'],
		'conditions': [
			[ 'OS=="linux" or OS=="freebsd" or OS=="openbsd" or OS=="solaris"',{
				'ldflags': [ '<!@(pkg-config --libs --libs-only-other opencv)' ],
				'libraries': [ '<!@(pkg-config --libs opencv)' ],
				'cflags': [ '<!@(pkg-config --cflags opencv)' ],
				'cflags_cc': [ '<!@(pkg-config --cflags opencv)' ],
				'cflags_cc!': ['-fno-rtti'],
				'cflags_cc+': ['-frtti']
			}]],
      "include_dirs": ['$(OPENCV_DIR)/include'],
      'library_dirs': ['$(OPENCV_DIR)/x64/vc12/lib'], 
      'libraries':    ['-lopencv_calib3d2410d.lib',
					 	'-lopencv_core2410d.lib',
										'-lopencv_features2d2410d.lib',
										'-lopencv_flann2410d.lib',
										'-lopencv_highgui2410d.lib',
										'-lopencv_imgproc2410d.lib',
										'-lopencv_ml2410d.lib',
										'-lopencv_gpu2410d.lib',
										'-lopencv_objdetect2410d.lib',
										'-lopencv_photo2410d.lib',
										'-lopencv_stitching2410d.lib',
										'-lopencv_superres2410d.lib',
										'-lopencv_ts2410d.lib',
										'-lopencv_video2410d.lib',
										'-lopencv_videostab2410d.lib'],
		 'msvs_settings': 
					    {
			          'VCCLCompilerTool': {
			            'RuntimeLibrary': 0, # static release
			            'Optimization': 3, # /Ox, full optimization
			            'FavorSizeOrSpeed': 1, # /Ot, favour speed over size
			            'InlineFunctionExpansion': 2, # /Ob2, inline anything eligible
			            'WholeProgramOptimization': 'true', # /GL, whole program optimization, needed for LTCG
			            'OmitFramePointers': 'true',
			            'EnableFunctionLevelLinking': 'true',
			            'EnableIntrinsicFunctions': 'true',
			            'RuntimeTypeInfo': 'false',
			            'ExceptionHandling': '0',
			            'AdditionalOptions': [
			              '/MP /EHsc', '/MTd'
			            ],
			          },
			          'VCLibrarianTool': {
			            'AdditionalOptions': [
			              '/LTCG', # link time code generation
			            ],
			          },
			          'VCLinkerTool': {
			            'LinkTimeCodeGeneration': 1, # link-time code generation
			            'OptimizeReferences': 2, # /OPT:REF
			            'EnableCOMDATFolding': 2, # /OPT:ICF
			            'LinkIncremental': 1, # disable incremental linking
			          	'AdditionalLibraryDirectories': ['$(OPENCV_ROOT)/x64/vc12/lib'],
			          }
			        }
    }
  ]
}