"""
Author: Smol
E-mail address: kingalexander471@gmail.com
Liscense: Creative Commons BY-SA

S3DR - Version 1.2.4
CHANGES:
Began the documentation process
Started making it into a functional library instead of a standalone script
"""
import pygame
import math as Math
import sys
pygame.font.init()
class Transformation():
    def __init__(self, mode, *args):
        self.mode=mode
        self.args=args
def loadOBJ(filename:str, color:pygame.Color, transform:list[Transformation]):
    '''
    ## loadOBJ:
    Returns an Model object from the given OBJ file

    - filename | A string representing the file path of the .obj file
    - color | A color type object (str, list, pygame.Color) that represents the base color assigned to a Model
    - transform | A list of global transformations to apply to the object every frame
    '''
    vertices:list[Point] = []
    faceData = []
    for line in open(filename, 'r'):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = Point(float(values[1]), float(values[2]), float(values[3]))
            vertices.append(v)
        elif values[0] == 'f':
            face = []
            for v_triplet in values[1:]:
                v_index = int(v_triplet.split('/')[0])
                face.append(v_index - 1) 
            faceData.append(tuple(face))
    return Model(
        vertices=vertices, 
        faces=faceData, 
        color=color,
        transform=transform
    )
def loadS3DM(filename:str, color:pygame.Color, transform:Transformation):
    '''
    ## loadS3DM:
    Returns an Model object from the given S3DM file

    - filename | A string representing the file path of the .s3dm file
    - color | A color type object (str, list, pygame.Color) that represents the base color assigned to a Model
    - transform | A list of global transformations to apply to the object every frame
    '''
    import numpy as np
    file = open(filename, "rb").read()
    if(not file[:5] == b"S3DM\n"):
        raise ValueError("The provided S3DM file is improperly formatted")
    vertices = []
    faces = []
    file = file[5:].split(b'\xFF\xFF\xFF\xFF\xFF\xFF')
    vertexArrays = [file[0][i:i+12] for i in range(0, len(file[0]), 12)]
    for n, vertex in enumerate(vertexArrays):
        vertex = [float(np.frombuffer(vertex[i:i+4], dtype=np.float32)[0]) for i in range(0, len(vertex), 4)]
        vertices.append(Point(vertex[0], vertex[1], vertex[2]))
    faceArrays = [file[1][i:i+9] for i in range(0, len(file[1]), 9)]
    for n, face in enumerate(faceArrays):
        face = [int.from_bytes(face[i:i+3], "little")-1 for i in range(0, len(face), 3)]
        faces.append(face)
    return Model(
        vertices,
        faces,
        color,
        transform=transform
    )
def keep(cond:function, array:list):
    result=[]
    for i in array:
        if(cond(i)):
            result.append(i)
    return result
class Point():
    def __init__(self, x:float|int, y:float|int, z:float|int=False):
        if(z or z==0):
            self.z=z
            self.d3=True
            self.matrix=Matrix([[x],[y],[z]])
            self.coords = (x,y,z)
        else:
            self.d3=False
            self.matrix=Matrix([[x],[y]])
            self.coords=(x,y)
        self.y=y
        self.x=x
    def __repr__(self) -> str:
        return (str(self.coords))
class Matrix():
    def __init__(self, values:list[list]):
        self.rows = len(values)
        self.columns = len(values[0])
        self.matrix = values
    
    def __mul__(self, other):
        if(not isinstance(other, Matrix)):
            raise ValueError(f"Cannot multiply types Matrix and {type(other)}")
        rowsA = self.rows
        colsA = self.columns
        rowsB = other.rows
        colsB = other.columns
        if (not colsA == rowsB):
            raise ValueError('Incompatible matrices for multiplication')
        result = [[0 for i in range(colsB)] for i in range(rowsA)]
        for i in range(rowsA):
            for j in range(colsB):
                for k in range(colsA):
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return Matrix(result)
    def __getitem__(self, index):
        return self.matrix[index]
    def __setitem__(self, index, value):
        self.matrix[index] = value
    def __delitem__(self, index):
        del self.matrix[index]
    def toPoint(self):
        if(self.columns>1 or self.rows>3):
            raise ValueError("Matrix cannot be converted to point: not a point-type matrix")
        if(self.rows==2):
            self = Point(self.matrix[0][0],self.matrix[1][0])
        else:
            self = Point(self.matrix[0][0],self.matrix[1][0],self.matrix[2][0])
        return self
    def __sub__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
             raise ValueError('Matrices must have the same dimensions for subtraction')
        result = []
        for i in range(self.rows):
            new_row = [self.matrix[i][0] - other.matrix[i][0]]
            result.append(new_row)
        return Matrix(result)
    def dot(self, other):
        result = 0
        for i in range(3):
            result += self.matrix[i][0] * other.matrix[i][0]
        return result
    def cross(self, other):
        a = self.matrix
        b = other.matrix
        x = a[1][0] * b[2][0] - a[2][0] * b[1][0]
        y = a[2][0] * b[0][0] - a[0][0] * b[2][0]
        z = a[0][0] * b[1][0] - a[1][0] * b[0][0]
        
        return Matrix([[x], [y], [z]])
    def normalize(self):
        if not (self.rows == 3 and self.columns == 1):
            raise ValueError("Normalization requires a 3x1 vector matrix.")

        x, y, z = self.matrix[0][0], self.matrix[1][0], self.matrix[2][0]
        length = (x*x + y*y + z*z)**0.5
        
        if length == 0:
            return Matrix([[0],[0],[0]])
            
        return Matrix([[x / length], [y / length], [z / length]])
    def __repr__(self) -> str:
        return str(self.matrix)
    def __add__(self, other):
        if(not isinstance(other, (Matrix, Point))):
            raise TypeError(f"Incorrect operands for '+' (Matrix and {type(other)}) | Can only add Matrix or Point to Matrix")
        elif(isinstance(other, Point)):
            other = other.matrix
        if(not self.rows == other.rows or not self.columns == other.columns):
             raise ValueError("'Matrix' objects  must have the same dimensions for addition")
        result = []
        for i in range(self.rows):
            row=[]
            for o in range(self.columns):
                row.append(self.matrix[i][o] + other.matrix[i][o])
            result.append(row)
        return(Matrix(result))
def rotPoint(point:Matrix|Point, angle:float|int=0, axis=False, center:Matrix|Point=Point(0,0,0), deltatime=1) -> Point:
    if(isinstance(axis, list)):
        center = axis[1]
        axis  = axis[0]
        if(len(axis)>2):
            deltatime = axis[2]
    if(callable(center)):
        center = center()
    if(callable(deltatime)):
        deltatime = deltatime()
    if(not isinstance(point, (Matrix,Point))):
        raise TypeError(f"'point' argument must be of type Matrix or Point, not of type {type(point).__repr__()}")
    elif(isinstance(point, Point)):
        point = point.matrix
    if(not isinstance(center, (Matrix,Point))):
        raise TypeError(f"'center' argument must be of type Matrix or Point, not of type {type(point).__repr__()}")
    elif(isinstance(center, Point)):
        center = center.matrix
    point = point-center
    angle = angle*deltatime
    if(point.rows==3):
        if(not axis):
            raise SyntaxError("No 'axis' argument supplied to rotPoint function when a 3d point was given")
        elif(axis=="x"):
            return (Matrix([
                [1,0,0],
                [0,Math.cos(Math.radians(angle)),-Math.sin(Math.radians(angle))],
                [0,Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle))]
            ])*point)+center
        elif(axis=="y"):
            return (Matrix([
                [Math.cos(Math.radians(angle)),0,Math.sin(Math.radians(angle))],
                [0,1,0],
                [-Math.sin(Math.radians(angle)),0,Math.cos(Math.radians(angle))]
            ])*point)+center
        elif(axis=="z"):
            return (Matrix([
                [Math.cos(Math.radians(angle)),-Math.sin(Math.radians(angle)),0],
                [Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle)),0],
                [0,0,1]
            ])*point)+center
        else:
            raise ValueError("'axis' argument for rotPoint function must be 'x', 'y', or 'z' when working with a 3d point")
    elif(point.rows==2):
        return Matrix([[Math.cos(Math.radians(angle)), -Math.sin(Math.radians(angle))],[Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle))]])*point
    else:
        raise ValueError("Given matrix is improperly formatted to be a 2d or 3d point")
def transformPoint(point:Matrix, mode:str, *args) -> Matrix:
    if(len(args)<2):
        args = args[0]
    if(mode=="rotate"):
        if(len(args)>2):
            return rotPoint(point, angle=args[0], axis=[args[1], args[2]])
        return rotPoint(point, args[0], args[1])
    elif(mode=="translate"):
        start = point.matrix
        if(args[0]):
            return Matrix([[start[0][0]+args[1][0]],[start[1][0]+args[1][1]],[start[2][0]+args[1][2]]])
        else:
            return Matrix([[start[0][0]+args[1][0]],[start[1][0]+args[1][1]]])
    elif(mode=="scale"):
        start = (point-args[0].matrix).matrix
        if(args[0].d3):
            return Matrix([[start[0][0]*args[1][0]],[start[1][0]*args[1][1]],[start[2][0]*args[1][2]]])+args[0]
        else:
            return Matrix([[start[0][0]*args[1][0]],[start[1][0]*args[1][1]]])+args[0]
def getCenter(verts:list[Point]) -> Point:
    x=0
    y=0
    z=0
    total = len(verts)
    for vertex in verts:
        x+=vertex.x
        y+=vertex.y
        z+=vertex.z
    x=x/total
    y=y/total
    z=z/total
    return(Point(x,y,z))
class Model():
    def __init__(self, vertices:list[Point], faces:list[int], color:pygame.Color, transform:Transformation):
        self.vertices = vertices
        self.vertices_DRAW = vertices
        self.faces = faces
        self.color = color
        self.transform=transform
        self.visible = True
    def moved(self, x:int|float, y:int|float,z:int|float):
        verts=[]
        for vert in self.vertices:
            verts.append(transformPoint(vert.matrix, "translate", [True, [x,y,z]]).toPoint())
        return Model(
            verts,
            self.faces,
            self.color,
            self.transform,
            self.objtype
        )
    def scaled(self, factor:int|float|list):
        if(isinstance(factor, (float, int))):
            factor = [factor, factor, factor]
        elif(not isinstance(factor, list)):
            raise TypeError(f"'factor' argument must be of type list, float, or int, not {type(factor)}")
        verts=[]
        for vert in self.vertices:
            verts.append(transformPoint(vert.matrix, "scale", [Point(0,0,0),factor]).toPoint())
        return Model(
            verts,
            self.faces,
            self.color,
            self.transform,
            self.objtype
        )
    def rotated(self, angle:float|int, axis:str, center:Point|Matrix=Point(0,0,0)):
        if(isinstance(center, Point)):
            center = center.matrix
        elif(not isinstance(center, (Matrix))):
            raise TypeError(f"'angle' argument must be Point or Matrix not {type(center)}")
        if(not isinstance(angle, (float, int))):
            raise TypeError(f"'angle' argument must be float or int not {type(angle)}")
        if(not isinstance(axis, (str))):
            raise TypeError(f"'angle' argument must be str not {type(axis)}")
        verts=[]
        for vertex in self.vertices:
            verts.append(transformPoint(vertex, "rotate", angle, [axis, center]).toPoint())
        return Model(
            verts,
            self.faces,
            self.color,
            self.transform,
            self.objtype
        )
    def update(self):
        self.center = getCenter(self.vertices)
class Player:
    vy=0
    drawnFaces=0
    camera=[0,0,0,0,0]
    drawMode="face"
    cull=True
    light = Matrix([[-80],[-150],[-80]])
    font=None
    fontTimer=1000
    dt=0
    totFaces = 0
global screen
import screeninfo
screen: pygame.Surface = pygame.display.set_mode((screeninfo.get_monitors()[0].width,screeninfo.get_monitors()[0].height), flags=pygame.FULLSCREEN)
pygame.display.set_caption("3D Renderer by Smol")
pygame.display.set_icon(pygame.image.load("C:\\Users\\kinal\\Downloads\\3DR by Smol 32x32.png"))
del screeninfo
Player.camera = [0,10,-200,0,0]
global camMatrix
camMatrix = Matrix([[Player.camera[0]],[Player.camera[1]],[Player.camera[2]]])

lis2tr=[Transformation("rotate", -1, ["z", lambda: objects[1].center]), Transformation("translate", True, [-1,0,0])]
global objects
objects: list[Model]=[
    Model(
    [
        Point(50, -50, 50),      # 0: (+X, +Y, +Z)
        Point(-50, -50, 50),     # 1: (-X, +Y, +Z)
        Point(-50, 50, 50),    # 2: (-X, -Y, +Z)
        Point(50, 50, 50),     # 3: (+X, -Y, +Z)
        Point(50, -50, -50),     # 4: (+X, +Y, -Z)
        Point(-50, -50, -50),    # 5: (-X, +Y, -Z)
        Point(-50, 50, -50),   # 6: (-X, -Y, -Z)
        Point(50, 50, -50),    # 7: (+X, -Y, -Z)
    ],
    [
        #(0,3,2,1),(4,5,6,7),(0,1,5,4),(3,7,6,2),(0,4,7,3),(1,2,6,5)
    ],
    "#ff0000",
    [],#Transformation("rotate", 0, "x"), Transformation("rotate", 0, "z"), Transformation("rotate", 0, "y")],
    "",
    ),
    loadS3DM("C:\\Users\\kinal\\Downloads\\chest.s3dm", "#ff8800", [Transformation("rotate", 0, "y")]).scaled(500).rotated(180, "x").moved(0,50,0),
]
for obj in objects:
    Player.totFaces+=len(obj.faces)
def draw(dt):
    background(screen, "black")
    drawModels(dt)
    Player.fontTimer+=1
    if(Player.fontTimer>=4):
        Player.font = pygame.font.Font("C:\\Windows\\Fonts\\consolab.ttf", 12).render(f"FPS: {str(round(clock.get_fps())).zfill(3)} | Faces: {str(Player.drawnFaces).zfill(len(str(Player.totFaces)))} ({f"{round((Player.drawnFaces/Player.totFaces)*100)}".zfill(3)}%) | X: {str(round(Player.camera[0])).zfill(3)}  Y: {str(round(Player.camera[1])).zfill(3)}  Z: {str(round(Player.camera[2])).zfill(3)} | Yaw: {str(round(Player.camera[3])).zfill(3)} | Pitch: {str(round(Player.camera[4])).zfill(3)} | Culling: {"On " if Player.cull else "Off"} | Platform: {sys.platform} | PythonVerson: {sys.version.split(" ")[0]} | Version: 1.2.1", False, "#ffffff")
        Player.fontTimer=0
    screen.blit(Player.font, (10,10))
    pygame.display.flip()

def faceTexture(obj, face_indices):
    sumU = 0.0
    sumV = 0.0
    for index in face_indices:
        u, v = obj.uv[index]
        sumU += u
        sumV += v
    avgU = sumU / len(face_indices)
    avgV = sumV / len(face_indices)
    width, height = obj.texture.get_size()
    texX = int(avgU) 
    texY = int(avgV)
    
    return obj.texture.get_at((texX, texY))

def drawModels(dt):
    Player.drawnFaces=0
    #Player.light = rotPoint(Player.light, 0.7, "y", Point(0,0,0))
    drawfaces=[]
    for obj in objects:
        projected=[]
        if(not obj.visible):
            continue
        for vertex in obj.vertices_DRAW:
            point = vertex.coords
            if(point[2]<=20):
                projected.append(None)
                continue
            zProjected = point[2]
            if(obj.objtype=="HUD"):
                zProjected = 1
            scale = 500/zProjected
            new_x = (point[0]*scale)+(screen.get_width()/2)
            new_y = (point[1]*scale)+(screen.get_height()/2)
            projected.append((new_x,new_y))
        for face in obj.faces:
            if(obj.texture):
                real_color= faceTexture(obj, face)
                real_color = "#"+hex(real_color.r)[2:].zfill(2)+hex(real_color.g)[2:].zfill(2)+hex(real_color.b)[2:].zfill(2)
            else:
                real_color = obj.color
            facecoords = []
            allvis=True
            for vert in face:
                proj_point = projected[vert]
                if proj_point == None:
                    allvis = False
                    break
                facecoords.append(proj_point)
            
            if not allvis:
                continue
            if Player.cull:
                if is_backface(facecoords):
                    continue
            v1, v2, v3 = (obj.vertices_DRAW[i] for i in face[:3])
            normal = getNormal(v1, v2, v3)
            diffuse_factor = normal.dot(Player.light.normalize())
            diffuse_factor = max(0.0, diffuse_factor) 
            light_intensity = min(0.25 + (1 * diffuse_factor), 1)
            final_color = shadeFace(real_color, light_intensity)
            x_depths = [obj.vertices_DRAW[i].coords[0] for i in face]
            #x_depth = sum(x_depths) / len(x_depths)
            y_depths = [obj.vertices_DRAW[i].coords[1] for i in face]
            #y_depth = sum(y_depths) / len(y_depths)
            z_depths = [obj.vertices_DRAW[i].coords[2] for i in face]
            #z_depth = sum(z_depths) / len(z_depths)
            depth = sum(z_depths) / len(z_depths)
            if((depth>5000 or depth<10) and Player.cull):
                continue
            drawfaces.append((depth, final_color, facecoords))
    drawfaces.sort(key=lambda x: x[0], reverse=True)
    for _, color, coords in drawfaces:
        if(Player.drawMode == "face"):
            pygame.draw.polygon(screen, color, coords)
        else:
            pygame.draw.polygon(screen, color, coords, 5)
    Player.drawnFaces = len(drawfaces)

def is_backface(projected_points):
    A, B, C = projected_points[0], projected_points[1], projected_points[2]
    winding_value = (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
    return winding_value > 0 

def getNormal(vert1: Point, vert2: Point, vert3: Point) -> Matrix:
    vec_a = vert2.matrix - vert1.matrix
    vec_b = vert3.matrix - vert1.matrix
    normal = vec_a.cross(vec_b).normalize()
    return normal

def hex2RGB(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def shadeFace(base_color_hex: str, light_intensity: float) -> tuple[int, int, int]:
    r, g, b = hex2RGB(base_color_hex)
    intensity = max(0.0, min(1.0, light_intensity))
    r = int(r * intensity)
    g = int(g * intensity)
    b = int(b * intensity)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

def background(sur:pygame.Surface, col:str):
    sur.fill(col)

def rotPoint(point:Matrix|Point, angle:float|int=0, axis=False, center:Matrix|Point=Point(0,0,0), deltatime=1):
    if(isinstance(axis, list)):
        center = axis[1]
        axis  = axis[0]
        if(len(axis)>2):
            deltatime = axis[2]
    if(callable(center)):
        center = center()
    if(callable(deltatime)):
        deltatime = deltatime()
    if(not isinstance(point, (Matrix,Point))):
        raise TypeError(f"'point' argument must be of type Matrix or Point, not of type {type(point)}")
    elif(isinstance(point, Point)):
        point = point.matrix
    if(not isinstance(center, (Matrix,Point))):
        raise TypeError(f"'center' argument must be of type Matrix or Point, not of type {type(point)}")
    elif(isinstance(center, Point)):
        center = center.matrix
    point = point-center
    angle = angle*deltatime
    if(point.rows==3):
        if(not axis):
            raise SyntaxError("No 'axis' argument supplied to rotPoint function when a 3d point was given")
        elif(axis=="x"):
            return (Matrix([
                [1,0,0],
                [0,Math.cos(Math.radians(angle)),-Math.sin(Math.radians(angle))],
                [0,Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle))]
            ])*point)+center
        elif(axis=="y"):
            return (Matrix([
                [Math.cos(Math.radians(angle)),0,Math.sin(Math.radians(angle))],
                [0,1,0],
                [-Math.sin(Math.radians(angle)),0,Math.cos(Math.radians(angle))]
            ])*point)+center
        elif(axis=="z"):
            return (Matrix([
                [Math.cos(Math.radians(angle)),-Math.sin(Math.radians(angle)),0],
                [Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle)),0],
                [0,0,1]
            ])*point)+center
        else:
            raise ValueError("'axis' argument for rotPoint function must be 'x', 'y', or 'z' when working with a 3d point")
    elif(point.rows==2):
        return Matrix([[Math.cos(Math.radians(angle)), -Math.sin(Math.radians(angle))],[Math.sin(Math.radians(angle)),Math.cos(Math.radians(angle))]])*point
    else:
        raise ValueError("Given matrix is improperly formatted to be a 2d or 3d point")

def inputHandling(dt):
    keys = pygame.key.get_pressed()
    moveSpeed = 400 * dt
    if(keys[pygame.K_LCTRL]):
        moveSpeed = 650 * dt
    rotSpeed = 90 * dt
    if(keys[pygame.K_w]):
        Player.camera[2]+=moveSpeed * Math.cos(Math.radians(Player.camera[3]))
        Player.camera[0]+=moveSpeed * Math.sin(Math.radians(Player.camera[3]))
    if(keys[pygame.K_s]):
        Player.camera[2]-= moveSpeed * Math.cos(Math.radians(Player.camera[3]))
        Player.camera[0]-= moveSpeed * Math.sin(Math.radians(Player.camera[3]))
    if(keys[pygame.K_a]):
        Player.camera[2]+=moveSpeed * Math.cos(Math.radians(Player.camera[3] - 90))
        Player.camera[0]+=moveSpeed * Math.sin(Math.radians(Player.camera[3] - 90))
    if(keys[pygame.K_d]):
        Player.camera[2]-=moveSpeed * Math.cos(Math.radians(Player.camera[3] - 90))
        Player.camera[0]-=moveSpeed * Math.sin(Math.radians(Player.camera[3] - 90))
    if(keys[pygame.K_LEFT]): 
        Player.camera[3] -=rotSpeed
    if(keys[pygame.K_RIGHT]):
        Player.camera[3] +=rotSpeed
    if(keys[pygame.K_UP]):
        Player.camera[4] +=rotSpeed
    if(keys[pygame.K_DOWN]):
        Player.camera[4] -=rotSpeed
    Player.camera[4] = max(Player.camera[4], -90)
    Player.camera[4] = min(Player.camera[4], 90)
    if(Player.camera[3]>360):
        Player.camera[3] = Player.camera[3]-360
    if(Player.camera[3]<0):
        Player.camera[3] = Player.camera[3]+360

    if(keys[pygame.K_SPACE] and Player.camera[1]>0):
        Player.vy+=5
        Player.camera[1]-=0.5
    if(Player.camera[1]>0):
        Player.camera[1]=0.1
        Player.vy=0
    else:
        Player.vy-=9.8*dt
    Player.camera[1]-=Player.vy

    if(keys[pygame.K_o]):
        Player.drawMode = "wire"
    else:
        Player.drawMode = "face"

    if(keys[pygame.K_p]):
        Player.cull = False
    else:
        Player.cull = True

def applyTransformation(obj:Model, mode, *args):
    tempverts=[]
    for vert in obj.vertices:
        tempverts.append(transformPoint(vert.matrix, mode, args).toPoint())
    obj.vertices = tempverts

def transformModels(dt):
    for obj in objects:
        if(obj.objtype == "updated"):
            obj.update()
        for transformation in obj.transform:
            applyTransformation(obj, transformation.mode, transformation.args[0], transformation.args[1])
        obj.vertices_DRAW=[]
        for vertex in obj.vertices:
            vertex = (vertex.matrix - camMatrix)
            vertex = rotPoint(vertex, -Player.camera[3], "y")
            vertex = rotPoint(vertex, -Player.camera[4], "x").toPoint()
            obj.vertices_DRAW.append(vertex)
        obj.visible = True
        #for vertex in obj.vertices_DRAW:
        #    if():
        #        obj.visible = True
        #        break

clock = pygame.time.Clock()

while True:
    for e in pygame.event.get():
        if e.type == 256:
            raise SystemExit
    dt = clock.tick(10000)/1000
    Player.dt = dt
    inputHandling(dt)
    transformModels(dt)
    draw(dt)
    camMatrix=Matrix([[Player.camera[0]],[Player.camera[1]],[Player.camera[2]]])
