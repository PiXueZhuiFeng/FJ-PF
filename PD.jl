include("graph.jl")


using LinearAlgebra
using Random
using Statistics
using Laplacians
using SparseArrays
using DataFrames
using IterativeSolvers
using SparseArrays
using Plots
using Base.Threads
 
using CSV

 

function sqrtsample(n::Int, p::Float64)
    result = Int[]
    i = 0
    while i <= n
        ii = floor(Int, log(rand()) / log(1 - p))
        i +=ii;
        if (ii != 0) && (i<=n)
            push!(result, i)
        end
    end
    return result
end
 


function PD_parallel(G, tau, c,Ev,alist,kk) 
    # G = deepcopy(G)
    S = Array{Tuple{Int32, Int32}, 1}()
    sev = size(Ev, 1)
    cho = zeros(Bool, sev)
    c = deepcopy(c)  
    d = [length(G.nbr[j]) for j in 1:G.n]             
    gnbr = deepcopy(G.nbr)
    a_threads = [Int[] for _ in 1:nthreads()]
    b_threads = [Int[] for _ in 1:nthreads()]
    InForest_pool = [falses(G.n) for _ in 1:nthreads()]
    Next_pool = [fill(-1, G.n) for _ in 1:nthreads()]
    Rootindex_pool = [zeros(Int, G.n) for _ in 1:nthreads()]
    for aaa = 1:kk 
        for i in 1:nthreads()
            empty!(a_threads[i])
            empty!(b_threads[i])
            fill!(InForest_pool[i], false)
            fill!(Next_pool[i], -1)
            fill!(Rootindex_pool[i], 0)
        end

        Threads.@threads for p in 1:tau
            thread_id = threadid()

            InForest = InForest_pool[thread_id]
            Next = Next_pool[thread_id]
            Rootindex = Rootindex_pool[thread_id]

            fill!(InForest, false)
            fill!(Next, -1)
            fill!(Rootindex, 0)

            cntnow = 0; 
            for i in alist
                u = i
                # cntnow += 1;
                while !InForest[u]
                    seed = rand()
                    if seed <= 1 / (1 + d[u])
                        InForest[u] = true
                        Rootindex[u] = u
                    else
                        k = floor(Int, seed * (1 + d[u]))
                        Next[u] = gnbr[u][k]
                        u = Next[u]
                        cntnow += 1;
                    end
                end

                rootnow = Rootindex[u]
                push!(a_threads[thread_id], i)
                push!(b_threads[thread_id], rootnow)

                u = i
                while Rootindex[u] == 0
                    InForest[u] = true
                    Rootindex[u] = rootnow
                    u = Next[u]
                end
            end

        end

        a = vcat(a_threads...)
        b = vcat(b_threads...)
        W = sparse(a,b,1.0,G.n,G.n)
        W = W./tau
        q = W * c

        dz = zeros(sev)
        
        for j = 1 : sev
            if cho[j]
                continue
            end
            (u, v) = Ev[j]
            dz[j] = (q[u] - q[v])^2 / (1.0 + W[u, u] + W[v, v] - 2*W[u, v])
        end
        xx = argmax(dz)
        cho[xx] = true
        (su, sv) = Ev[xx]
        # W = updateW(W, su, sv)
        d[su] += 1
        d[sv] +=1
        push!(gnbr[su],sv)
        push!(gnbr[sv],su)
        push!(S, Ev[xx])
    end
    return S
end


 
 
function random_edgeset(L,G, k) 
    edges = Set{Tuple{Int, Int}}()
    while length(edges) < k
        u = rand(1:G.n)
        v = rand(1:G.n)
        if u != v && !in((u, v), edges) && !in((v, u), edges) && L[u,v] == 0
            push!(edges, (u, v))
        end
    end
    edges_array = collect(edges)
    return edges_array
end

function exactopinion(G,s,Ev,k)
    s = deepcopy(s)  
    W = inv(Matrix(I + lapsp(G)))
    S = Array{Tuple{Int32, Int32}, 1}()
    sev = size(Ev, 1)
    cho = zeros(Bool, sev)
    for i = 1 : k
        dz = zeros(sev)
        q = W * s
        for j = 1 : sev
            if cho[j]
                continue
            end
            (u, v) = Ev[j]
            dz[j] = (q[u] - q[v])^2 / (1.0 + W[u, u] + W[v, v] - 2*W[u, v])
        end
        xx = argmax(dz)
        cho[xx] = true
        (su, sv) = Ev[xx]
        W = updateW(W, su, sv)
        push!(S, Ev[xx])
    end
    return S
end

function updateW(W, u, v)
    c = W[:,u] - W[:,v]
    fac = 1.0 / (1.0 + c[u] - c[v])
    return W - fac * c * c'
end




function approxOpinion(G, s, Ev, k; eps = 0.3)
    
    IpL = lapsp(G) +I
    B = Bsp(G)
    m = G.m
    n = G.n
    kkk = round(Int, 0.5 * log2(n) / (eps^2))
    kkk = 10
    S = Array{Tuple{Int32, Int32}, 1}()
    sev = size(Ev, 1)
    cho = zeros(Bool, sev)
    for rep = 1 : k
        println("approxopinion",rep)
        dzfm = zeros(sev)
        f =  approxchol_sddm(IpL)
        for i = 1 : kkk
            yy1 = B' * randn(m)
            yy2 = randn(n)
            zz1 = f(yy1)
            zz2 = f(yy2)
            for j = 1 : sev
                (uu, vv) = Ev[j]
                dzfm[j] += ((zz1[uu] - zz1[vv])^2 + (zz2[uu] - zz2[vv])^2)
            end
        end
        dzfm ./= kkk
        dzfm .+= 1.0
        q = f(s)

        dz = zeros(sev)
        for j = 1 : sev
            if cho[j]
                continue
            end
            (u, v) = Ev[j]
            dz[j] = (q[u] - q[v])^2 / dzfm[j]
        end
        xx = argmax(dz)
        cho[xx] = true
        (su, sv) = Ev[xx]
        IpL[su, su] += 1
        IpL[sv, sv] += 1
        IpL[su, sv] = -1
        IpL[sv, su] = -1
        push!(S, Ev[xx])
    end
    return S
end

function score(G, addE, s)
    s = deepcopy(s)
    L = lapsp(G) + I
    f = approxchol_sddm(L)
    ans = [s'*f(s)]
    for (u, v) in addE
        L[u,v] -=1
        L[u,u] +=1 
        L[v,u] -=1
        L[v,v] +=1 
        f = approxchol_sddm(L)
        push!(ans,s'*f(s))
        println("score ",ans[end])
    end
    return ans
end

function randomSelect(Ev, k; randomSeed = round(Int, time() * 10000))
    Random.seed!(randomSeed)
    y = randperm(size(Ev, 1))
    S = Array{Tuple{Int32, Int32}, 1}()
    for i = 1 : k
        push!(S, Ev[y[i]])
    end
    return S
end

