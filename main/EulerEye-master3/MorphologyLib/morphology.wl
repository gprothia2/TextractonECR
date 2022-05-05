(* ::Package:: *)

(* ::Title:: *)
(*Morphological Process*)


(* ::Section:: *)
(*Morphological Operators to Build Connected Components *)


(* ::Subsection:: *)
(*Erode, Dilate, Open, Close*)


erode[image_, ker_, r_]:= Nest[Erosion[#, ker]&, image, r]
dilate[image_, ker_, r_]:= Nest[Dilation[#, ker]&, image, r]
open[image_, ker_, r_]:= dilate[erode[image, ker, r], ker, r]
close[image_, ker_, r_]:= erode[dilate[image, ker, r], ker, r]


(* ::Subsection:: *)
(* Operator Kernel *)


kernelFromStructure[structure_] := Block[
  {size = 2 Max @ Abs @ MinMax @ structure + 1},
  Transpose @ ReplacePart[
    Table[0, size, size],
    Thread[Append[structure, {0, 0}] + (size + 1)/2 -> 1]
  ]
]
linearStrut[kernelSize_] := Table[
  Round[{Cos[t],Sin[t]}*#] &/@ (Range[kernelSize]-(Floor[kernelSize/2]+1)),
  {t, 0, 11\[Pi]/12, \[Pi]/24}]

horizontalKernel[size_]:= kernelFromStructure[linearStrut[size][[1]]]
verticalKernel[size_]:= kernelFromStructure[linearStrut[size][[-11]]]

structSquare = {{-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}};
structCross = {{0,1}, {0,-1}, {1,0}, {-1,0}};

{crossKer, squareKer} = kernelFromStructure /@ {structCross,structSquare};


(* ::Section:: *)
(* Build Connected "Island" *)


(* ::Subsection:: *)
(*Label Components*)


labelComponents[img_, conn_]:=Module[
  {labels, sizes, s, width, height, ct, inds, queue, mask, qx, qy, nx, ny},
  (* Initialize *)
  {width,height}=Dimensions[img];
  labels=Table[0,{width},{height}];
  ct=0;
  If[conn==0,mask=structSquare,mask=structCross];
  (* Loop over all unvisited foreground pixels *)
  Do[
    If[img[[x,y]]==1&&labels[[x,y]]==0,
      ct++;
      (* start flooding from x,y *)
      queue={{x,y}};
      labels[[x,y]]=ct;
      While[queue!={},
        {qx,qy}=First[queue];
        queue=Drop[queue,1];
        Scan[
          {{nx,ny}={qx,qy}+#;
          If[0<nx<=width&&0<ny<=height&&img[[nx,ny]]==1&&labels[[nx,ny]]==0,
            AppendTo[queue,{nx,ny}];
            labels[[nx,ny]]=ct]}&,
          mask]];
    ],
    {x,width},{y,height}];
  labels]



(* ::Subsection:: *)
(*Get Largest Components*)


getIslandSizes[imgData_, conn_]:= Module[
	{labels = labelComponents[imgData, conn]},
	Table[Count[labels, i, 2], {i, 1, Max[labels]}];
]


getLargestComponents[img_,k_,conn_]:= Module[
	{labels = labelComponents[img, conn], sizes, inds},
	sizes = Table[Count[labels, i, 2], {i, 1, Max[labels]}];
	inds = Ordering[sizes, -Min[k, Length[sizes]]];
	Map[Map[If[MemberQ[inds,#], 1, 0] &, #]&, labels]
];


getKLargestComponets[binaryImg_, componentSize_, connType_]:= Block[
	{labels = labelComponents[binaryImg, connType], k, sizes, inds},
	sizes = Table[Count[labels, i, 2], {i, 1, Max[labels]}];
	k = Length[Select[sizes, # > componentSize &]];
	Print[k];
	inds = Ordering[sizes, -Min[k, Length[sizes]]];
	Map[Map[If[MemberQ[inds,#], 1, 0] &, #]&, labels]
]


(* ::Subsection:: *)
(*Morphological Process for BBZ*)


(* ::Subsubsection::Closed:: *)
(*Version 1*)


morphologicalProcessBBZv1[maskImg_]:= Module[
	{threshold, binaryImg, b1, b2, b3, b4, b5},
	(* 1. Binarize the given grayscale mask with auto-thresholding technique *)
	threshold = FindThreshold[maskImg];
	binaryImg = Binarize[maskImg, threshold];
	(* 2. Apply Morphological Operators *)
	(* 2.1: Fill in the small holes in this image *)
	b1 = close[maskImg, crossKer, 2];
	(* 2.2: Taking largest 2 components *)
	b2 = getLargestComponents[1 - ImageData[b1], 1, 0];
	(* 2.3: ?*)
	b3 = open[Image[b2], verticalKernel[3], 3];
	b4 = getLargestComponents[1 - ImageData[b3], 1, 1];
	b4
]


(* ::Subsubsection:: *)
(*Version 2*)


morphologicalProcessBBZv2[maskImg_, size_]:= Module[
	{threshold, binaryImg, b0, b0Vertical, b1, b1Labels},
	(* 1. Binarize the given grayscale mask with auto-thresholding technique *)
	threshold = FindThreshold[maskImg];
	binaryImg = Binarize[maskImg, threshold];
	(* 2. Apply Morphological Operators *)
	(* 2.1: Separating headers with the contents *)
	b0 = open[binaryImg, squareKer, 1];
	b0Vertical = close[b0, verticalKernel[3], 2];
	(* 2.2: Remove background noises *)
	b1 = getKLargestComponets[ImageData[b0Vertical], size, 0];
	b1Labels = labelComponents[b1, 0]
]


(* ::Subsection:: *)
(*Connected Components to Bounding Box*)


getSingleMask[maskLabels_, label_]:= Module[
	{numLabels, complement, substitutionRule, singleMask},
	numLabels = Union @ Flatten @ maskLabels;
	complement = Complement[numLabels, {0, label}];
	substitutionRule = Sort @ Flatten[
		{Thread[complement -> Table[0, Length[complement]]], {label -> 1}}
	];
	singleMask = maskLabels /. substitutionRule
]


convertAnnotationMask[vtx_List]:=
	Map[Transpose[{0,512}+{1,-1} Transpose[Partition[#,2]]]&, vtx]

getBoundingBox[singleMask_]:= Module[
	{positions, xMinMax, yMinMax},
	positions = Position[singleMask, 1];
	yMinMax = MinMax[positions[[All, 1]]];
	xMinMax = MinMax[positions[[All, 2]]];
	Flatten[convertAnnotationMask @ Transpose[{xMinMax, yMinMax}], 1]
]

getBoundingBoxes[maskLabels_]:= Module[
	{labels},
	labels = Rest @ Union @ Flatten @ maskLabels;
	getBoundingBox[getSingleMask[maskLabels, #]] &/@ labels
]


(* ::Subsubsection:: *)
(*Convert Tensor's Bounding Box Back To Original Image Size*)


convertTensorBBoxesToOriginalImageSingle[bbox_, {w_,h_}]:= Module[
	{ratioW, ratioH, xMin, xMax, yMin, yMax, newXmin, newXmax, newYmin, newYmax},
	{xMin, yMin, xMax, yMax} = Flatten @ bbox;
	ratioW = w / 512;
	ratioH = h / 512;
	newXmin = Round[xMin * ratioW];
	newXmax = Round[xMax * ratioW]; 
	newYmin = Round[yMin * ratioH];
	newYmax = Round[yMax * ratioH]; 
	{{newXmin, newYmin}, {newXmax, newYmax}}
]


convertTensorBBoxesToOriginalImage[bboxes_, {w_, h_}] := 
	convertTensorBBoxesToOriginalImageSingle[#, {w, h}] &/@ bboxes


(* ::Section:: *)
(* Visualization *)


(* ::Subsection:: *)
(*Dynamic Morphological Operation Visualization*)


morphologicalDynamic[input_]:=
    Manipulate[
      Evaluate[
        Switch[
          operator,
          "Erode", erode,
          "Dilate", dilate,
          "Open", open,
          "Close", close
        ]
        [
          Switch[
            image,
            "img-1", input[[1]],
            "img-2", input[[2]],
            "img-3", input[[3]],
            "img-4", input[[4]],
            "img-5", input[[5]]
          ]
          ,
          Switch[
            element,
            "Vertical", verticalKernel[kernelSize],
            "Horizontal", horizontalKernel[kernelSize],
            "8-connected", squareKer,
            "4-connected", crossKer
          ],
          t
        ]
      ],
      {t, 0, 15, 1},
      {kernelSize, 3, 50, 4},
      {operator, {"Erode", "Dilate", "Open", "Close"}},
      {element, {"Vertical", "Horizontal", "8-connected", "4-connected"}},
      {image, {"img-1", "img-2", "img-3", "img-4", "img-5"}}
    ]

getRGBValue[color_]:= color /. RGBColor -> List
visualizeMaskBlocks[maskLabels_]:= Module[
  {rc, colorMap, numLabels},
  numLabels = Union @ Flatten@maskLabels;
  rc = RandomColor[Length[numLabels]];
  colorMap = Thread[numLabels -> getRGBValue/@rc];
  Image[maskLabels /. colorMap]
]


visualizeSingleMask[maskLabels_, n_]:=
	Manipulate[Image @ getSingleMask[maskLabels, i], {i, 1, n, 1}]


(* ::Subsection:: *)
(*Plot Bounding Boxes*)


plotBBox[bbox_]:= Graphics[
	{EdgeForm[Directive[Thick,RandomColor[]]], White, Opacity[0.3], Rectangle@@@bbox}]

showBBox[img_, bbox_]:= Block[ 
	{bboxPlot}, 
	bboxPlot = plotBBox[bbox];
	Show[img, bboxPlot]	
]
