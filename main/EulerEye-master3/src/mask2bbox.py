"""
mask2bbox.py
----------------
Convert fcn probability mask into bounding boxes in Mathematica

@author Zhu, Wenzhen (wenzhu@amazon.com)
"""
# Import Wolfram Engine Packages
# To make this step working, we will need to set up Wolfram Engine.
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr


def mask_2_bboxes(mask_file_name, w, h):
    """
    :param mask_file_name:
    :param w: int
    :param h: int
    :param opt: 1,2 means dataset 1 or 2, which leads to the ocr results has coordinates or not
    :return: the full bbox file name path as string
    """
    wl_expr = 'erode[image_, ker_, r_]:= Nest[Erosion[#, ker]&, image, \
r];dilate[image_, ker_, r_]:= Nest[Dilation[#, ker]&, image, \
r];open[image_, ker_, r_]:= dilate[erode[image, ker, r], ker, \
r];close[image_, ker_, r_]:= erode[dilate[image, ker, r], ker, \
r];kernelFromStructure[structure_] := Block[{size = 2 Max @ Abs @ \
MinMax @ structure + 1},Transpose@ReplacePart[Table[0, size, size], \
Thread[Append[structure, {0, 0}] + (size + 1)/2 -> \
1]]];linearStrut[kernelSize_] := Table[Round[{Cos[t],Sin[t]}*#] \
&/@(Range[kernelSize]-(Floor[kernelSize/2]+1)), \
             {t, 0, 11\[Pi]/12, \[Pi]/24}];horizontalKernel[size_]:= \
kernelFromStructure[linearStrut[size][[1]]];verticalKernel[size_]:= \
kernelFromStructure[linearStrut[size][[-11]]];structCross = {{-1,-1}, \
{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}};structSquare = \
{{0,1}, {0,-1}, {1,0}, {-1,0}};{squareKer, crossKer} = \
kernelFromStructure /@ {structCross, structSquare}; \
\
\
morphologicalProcess[img_, thres_, "moreBbx"]:= \
Block[{res1, bin},bin = Binarize[img, thres];res1 = open[bin, \
squareKer, 2];open[res1, horizontalKernel[3], 14]];neighborMask4 = \
structCross;neighborMask8 = \
structSquare;labelComponents[img_,conn_]:=Module[{labels,sizes,s,\
width,height,ct,inds,queue,mask,qx,qy,nx,ny},(* Initialize \
*){width,height}=Dimensions[img];labels=Table[0,{width},{height}];ct=\
0;If[conn==0,mask=neighborMask8,mask=neighborMask4];(* Loop over all \
unvisited foreground pixels \
*)Do[If[img[[x,y]]==1&&labels[[x,y]]==0,ct++;(* start flooding from \
x,y *)queue={{x,y}};labels[[x,y]]=ct;While[queue!={},{qx,qy}=First[\
queue];queue=Drop[queue,1];Scan[{{nx,ny}={qx,qy}+#;If[0<nx<=width&&0<\
ny<=height&&img[[nx,ny]]==1&&labels[[nx,ny]]==0,AppendTo[queue,{nx,ny}\
];labels[[nx,ny]]=ct]}&,mask]];],{x,width},{y,height}];labels];flood[\
img_, conn_]:= Module[{labels, queue, qx, qy, nx, ny, width, height, \
mask, visited, count},(* Initialize *) {width, height} = \
Dimensions[img];labels = Table[0, {width}, {height}]; visited = \
Table[0, {width}, {height}]; If[conn == 0,mask = neighborMask8, mask = \
neighborMask4];visited[[1,1]] = 1; labels[[1,1]] = 1; count = \
0;Table[If[img[[i,j]] == 1 && visited[[i,j]] == 0,visited[[i,j]] = \
1;(*bfs[x, y, img, visited];*) queue = {{i, j}}; count = count + \
1; While[queue != {}, (*queue.pop()*) {qx,qy} = First[queue]; queue = \
Drop[queue, 1]; Scan[{{nx, ny} = {qx, qy} + #; If[0< nx <= width && 0 \
< ny <= height && img[[nx,ny]] == 1 && visited[[nx,ny]] == \
0, AppendTo[queue, {nx, ny}];        visited[[nx,ny]] = 1;        \
labels[[nx, ny]] = count;]} &,mask]];],{i, width},{j, height}];(* \
Return *)labels];getSingleMask[maskLabels_, label_]:= \
Module[{numLabels, complement, substitutionRule, \
singleMask},numLabels = Union @ Flatten @ maskLabels;complement = \
Complement[numLabels, {0, label}];substitutionRule = Sort @ \
Flatten[{Thread[complement -> Table[0, Length[complement]]], {label -> \
1}}];singleMask = maskLabels /. substitutionRule];padZero[k_Integer] := \
StringPadLeft[ToString[k], 4, "0"];getBoundingBox[singleMask_, w_, \
h_]:= Module[{positions, xMin, xMax, yMin, yMax, ratioW, ratioH, \
newXmin, newYmin, newXmax, newYmax},positions = Position[singleMask, \
1];{yMin, yMax} = MinMax[positions[[All, 1]]];{xMin, xMax} = \
MinMax[positions[[All, 2]]];(*Print[{xMin, xMax, yMin, \
yMax}];*)ratioW = w / 512;ratioH = h / 512;newXmin = Round[xMin * \
ratioW];newXmax = Round[xMax * ratioW]; newYmin = Round[yMin * \
ratioH];newYmax = Round[yMax * ratioH]; {newXmin, newYmin, newXmax, \
newYmax}];getBoundingBoxes[maskLabels_, w_, \
h_]:=Module[{labels},labels = Rest @ Union @ Flatten @ \
maskLabels;getBoundingBox[getSingleMask[maskLabels, #], w, h] &/@ \
labels];getBBoxes[mask_, thes_, w_, h_]:= Module[    {morphoMask, \
maskLabels, morphoMaskData, bboxes},morphoMask = \
morphologicalProcess[mask, thes, "moreBbx"];morphoMaskData = \
ImageData[morphoMask];(*we label components by \
4-connection*)maskLabels = \
labelComponents[morphoMaskData,1];getBoundingBoxes[maskLabels, w, \
h]];\
getBBoxAutomateThresh[maskPath_, {w_, h_}] :=  Module[{predMaskImg, thresh, box, boxDir, imgName},\
predMaskImg = Image[Import[maskPath]];   \
thresh = FindThreshold[predMaskImg] + 0.2; \
box = getBBoxes[predMaskImg, thresh, w, h]; \
boxDir = StringReplace[maskPath, "pred_mask" -> "bbox_raw"];\
Export[boxDir, box];\
boxDir];  \
'
    MMA_script = 'getBBoxAutomateThresh["%s", {%s, %s}]' % (mask_file_name, w, h)
    wolfram_script = wl_expr + MMA_script
    kernel = "/usr/local/Wolfram/WolframEngine/12.0/Executables/WolframKernel"
    session = WolframLanguageSession(kernel=kernel)
    res = session.evaluate(wlexpr(wolfram_script))
    session.terminate()
    return res
