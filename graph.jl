struct Graph
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int, 1}
    v :: Array{Int, 1} # uv is an edge
    # w :: Array{Array{Float64, 1}, 1} # weight of each edge
    nbr :: Array{Array{Int, 1}, 1}
    directed :: Bool
    # weighted :: Bool
end

mutable struct Graph1
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int, 1}
    v :: Array{Int, 1} # uv is an edge
    # w :: Array{Array{Float64, 1}, 1} # weight of each edge
    nbr :: Array{Array{Int, 1}, 1}
    directed :: Bool
    # weighted :: Bool
end

using SparseArrays

function get_graph(ffname, directed = false, dir = "data/")
    n = 0;
    Label = Dict{Int32, Int32}();
    Origin = Dict{Int32, Int32}();
    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1
    fname = string(dir,ffname);
    fin = open(fname, "r");
    str = readline(fin);
    str = split(str);
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3]);
    u = Int[];
    v = Int[];

    tot = 0
    for i = 1 : m
        str = readline(fin);
        str = split(str);
        x   = parse(Int, str[1]);
        y   = parse(Int, str[2]);
        if x!=y
            u1 = getID(x);
            v1 = getID(y);
            Origin[u1] = x;
            Origin[v1] = y;
            push!(u, u1);
            push!(v, v1);
            tot += 1;
        end
    end
    nbr=[ [ ] for i in 1:n ];
    for i=1:tot
        u1=u[i];
        v1=v[i];
        if directed
            push!(nbr[u1],v1);
        else
            push!(nbr[u1],v1);
            push!(nbr[v1],u1);
        end
    end

    close(fin)
    return Graph(n, tot, u, v, nbr,directed)
end


function get_graph1(ffname, directed = false, dir = "data/")
    n = 0;
    Label = Dict{Int32, Int32}();
    Origin = Dict{Int32, Int32}();
    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1
    fname = string(dir,ffname);
    fin = open(fname, "r");
    str = readline(fin);
    str = split(str);
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3]);
    u = Int[];
    v = Int[];

    tot = 0
    for i = 1 : m
        str = readline(fin);
        str = split(str);
        x   = parse(Int, str[1]);
        y   = parse(Int, str[2]);
        if x!=y
            u1 = getID(x);
            v1 = getID(y);
            Origin[u1] = x;
            Origin[v1] = y;
            push!(u, u1);
            push!(v, v1);
            tot += 1;
        end
    end
    nbr=[ [ ] for i in 1:n ];
    for i=1:tot
        u1=u[i];
        v1=v[i];
        if directed
            push!(nbr[u1],v1);
        else
            push!(nbr[u1],v1);
            push!(nbr[v1],u1);
        end
    end

    close(fin)
    return Graph1(n, tot, u, v, nbr,directed)
end

function get_graphnew(ffname, directed = false, dir = "data/")
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    getID(x::Int) = haskey(Label, x) ? Label[x] : (Label[x] = (n += 1))

    fname = string(dir, ffname)
    fin = open(fname, "r")
    str = readline(fin)
    str = split(str)
    m = parse(Int, str[3])

    u = Int[]
    v = Int[]
    edge_set = Set{Tuple{Int, Int}}()

    tot = 0
    for i = 1:m
        str = readline(fin)
        str = split(str)
        x = parse(Int, str[1])
        y = parse(Int, str[2])

        if x != y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y

            if directed
                push!(u, u1)
                push!(v, v1)
                tot += 1
            else
                # 对于无向图，保证 (u1, v1) 和 (v1, u1) 只存储一次
                edge = u1 < v1 ? (u1, v1) : (v1, u1)
                if !in(edge, edge_set)
                    push!(edge_set, edge)
                    push!(u, edge[1])
                    push!(v, edge[2])
                    tot += 1
                end
            end
        end
    end

    nbr = [Int[] for _ in 1:n]
    for i = 1:tot
        u1 = u[i]
        v1 = v[i]
        if directed
            push!(nbr[u1], v1)
        else
            push!(nbr[u1], v1)
            push!(nbr[v1], u1)
        end
    end

    close(fin)
    return Graph(n, tot, u, v, nbr, directed)
end

function lapsp(G )
	d = [length(G.nbr[j]) for j = 1:G.n];
	a=zeros(G.n);
	for i=1:G.n
		a[i]=i;
	end
	if G.directed
		uu=zeros(G.m+G.n);
		vv=zeros(G.m+G.n);
		ww=zeros(G.m+G.n);
		tot =1;
		for i = 1 :G.n
			for j = 1:length(G.nbr[i])
				uu[tot] = i;
				vv[tot] = G.nbr[i][j];
				ww[tot] = -1;
				tot += 1;
			end
		end

		uu[G.m+1:G.m+G.n]=a;
		vv[G.m+1:G.m+G.n]=a;
		ww[G.m+1:G.m+G.n]=d;
	    return sparse(uu,vv,ww)
	else
		uu=zeros(2*G.m+G.n);
		vv=zeros(2*G.m+G.n);
		ww=zeros(2*G.m+G.n);
		tot =1;
		for i = 1 :G.n
			for j = 1:length(G.nbr[i])
				uu[tot] = i;
				vv[tot] = G.nbr[i][j];
				ww[tot] = -1;
				tot += 1;
			end
		end
		uu[2*G.m+1:2*G.m+G.n]=a;
		vv[2*G.m+1:2*G.m+G.n]=a;
		ww[2*G.m+1:2*G.m+G.n]=d;
	    return sparse(uu,vv,ww)
	end
end

function Bsp(G::Graph)
    row = collect(1:G.m)
    col = [G.u; G.v]
    val = [ones(G.m); -ones(G.m)]
    B = sparse(vcat(row, row), col, val, G.m, G.n)
    return B
end 

function get_graphbig(ffname,m, directed = false,  dir = "data/")
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    getID(x::Int) = haskey(Label, x) ? Label[x] : (Label[x] = (n += 1))

    fname = string(dir, ffname)
    fin = open(fname, "r")
    str = readline(fin)
    str = split(str)
    # m = parse(Int, str[3])

    u = Int[]
    v = Int[]
    edge_set = Set{Tuple{Int, Int}}()

    tot = 0
    for i = 1:m
        if i % 10000000 == 0
            println(i/m)
        end
        if i == 1
            continue  # 跳过第一行
        end
        str = readline(fin)
        str = split(str)
        x = parse(Int, str[1])
        y = parse(Int, str[2])

        if x != y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y

            if directed
                push!(u, u1)
                push!(v, v1)
                tot += 1
            else
                # 对于无向图，保证 (u1, v1) 和 (v1, u1) 只存储一次
                edge = u1 < v1 ? (u1, v1) : (v1, u1)
                if !in(edge, edge_set)
                    push!(edge_set, edge)
                    push!(u, edge[1])
                    push!(v, edge[2])
                    tot += 1
                end
            end
        end
    end

    nbr = [Int[] for _ in 1:n]
    for i = 1:tot
        u1 = u[i]
        v1 = v[i]
        if directed
            push!(nbr[u1], v1)
        else
            push!(nbr[u1], v1)
            push!(nbr[v1], u1)
        end
    end

    close(fin)
    return Graph(n, tot, u, v, nbr, directed)
end