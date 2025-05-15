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

println("Number of threads: ", nthreads())


function rw(i,tau,s,d)
    z = 0;
    for p = 1:tau
        u = i;
        while true;
            seed = rand()
            if seed <= 1 / (1 + d[u])
                z += s[u];
                break
            else
                k = floor(Int, seed * (1 + d[u]));
                nextu = G.nbr[u][k];
                u = nextu;
            end
        end
    end
    z = z/tau;
    return z
end

# 





function lazy(G,i,l,r,s,d)
    z = 0;
    for p = 1:l
        u = i;
        for t = 1:r
            seed = rand();
            z += s[u]/(d[u]+1)

            if seed<=1/2
                continue
            elseif seed >= (1/2+d[u]/(2*(1+d[u])))
                break
            else
                k = floor(Int, (seed - 1/2)*(2*(1+d[u])) ) + 1;
                nextu = G.nbr[u][k];
                u = nextu;
            end
        end
    end
    z = z/(2*l);
    return z
end

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




function sqrtzzpre(G,tau,pp,s)  
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    a = Int[]
    b = Float64[]
    alist = sqrtsample(G.n, pp)
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        for i in alist
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    Rootindex[u] = u;
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            push!(a,i);
            push!(b,s[rootnow])
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow;             
                u = Next[u];
            end
        end
    end
    b = b/tau;
    Q = sparsevec(a,b,G.n)
    return Q
end


function rfz(G,tau,s) #algorithm SCF
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    z = zeros(G.n)
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        for i = 1:G.n
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    z[u] += s[u]
                    Rootindex[u] = u;
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow; 
                z[u] += s[rootnow]
                u = Next[u];
            end
        end
    end
    return z./tau
end



function absorbedrw(G,tau,s,d,pp)
    a = Int[];
    b = Float64[];
    alist =   sqrtsample(G.n, pp)
    for i in alist
        z = 0;
        for p = 1:tau
            u = i;
            while true;
                seed = rand()
                if seed <= 1 / (1 + d[u])
                    z += s[u];
                    break
                else
                    k = floor(Int, seed * (1 + d[u]));
                    nextu = G.nbr[u][k];
                    u = nextu;
                end
            end
        end
        z = z/tau;
        
        push!(a,i)
        push!(b,z)
    end
    return sparsevec(a,b,G.n)
end

function lazyz(G,l,r,s,d,pp)
    a = Int[];
    b = Float64[];
    alist =   sqrtsample(G.n, pp)
    for i in alist
        c = lazy(G,i,l,r,s,d)
        push!(a,i)
        push!(b,c)
    end
    return sparsevec(a,b,G.n)
end


function rerror(x,y)
    return norm(x-y)/norm(y)
end

function fjerror(z1,z2,s)  
    errors = []

    row_indices = rowvals(z2)
    abs_errors = [abs(z1[i] - z2[i])/z1[i] for i in row_indices]
    mean_abs_error = mean(abs_errors)
    push!(errors, mean_abs_error)


    barz1 = mean(z1)
    P1 = (z1.-barz1)'*(z1.-barz1)
    barz2 = sum(z2)/nnz(z2)
    row_indices = rowvals(z2)
    P2 = 0
    for i in row_indices
        P2 += (z2[i] -barz2)^2
    end
    P2 = P2/nnz(z2)*G.n
    push!(errors, rerror(P2, P1))

    I1 = (z1-s)'*(z1-s)
    I2 = 0
    for i in row_indices
        I2 += (z2[i] -s[i])^2
    end
    I2 = I2/nnz(z2)*G.n   
    push!(errors, rerror(I2, I1))

    C1 = z1'*z1
    C2 = 0
    for i in row_indices
        C2 += (z2[i])^2
    end
    C2 = C2/nnz(z2)*G.n   
    push!(errors, rerror(C2, C1))

    DC1 = z1'*s
    DC2 = 0
    for i in row_indices
        DC2 += z2[i]*s[i]
    end
    DC2 = DC2/nnz(z2)*G.n   
    push!(errors, rerror(DC2, DC1))

    D1 = DC1 - C1
    D2 = DC2 - C2
    push!(errors, rerror(D2, D1))

    return errors
end


function sqrtzzpre_partial_reset(G, tau, pp, s)
    d = [length(G.nbr[j]) for j in 1:G.n]
    InForest = falses(G.n)
    Next = fill(-1, G.n)
    Rootindex = zeros(Int, G.n)
    alist = sqrtsample(G.n, pp)

    a = Int[]
    b = Float64[]
    
    modified_nodes = Int[]           
    visited = falses(G.n)             

    for p in 1:tau

        if !isempty(modified_nodes)
            InForest[modified_nodes] .= false
            Next[modified_nodes] .= -1
            Rootindex[modified_nodes] .= 0
            visited[modified_nodes] .= false 
        end
        empty!(modified_nodes) 

        for i in alist
            u = i
            if !visited[u]
                push!(modified_nodes, u)
                visited[u] = true
            end

            while !InForest[u]
                seed = rand()
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true
                    Rootindex[u] = u
                else
                    k = floor(Int, seed * (1 + d[u]))
                    Next[u] = G.nbr[u][k]
                    u = Next[u]
                end

                if !visited[u]
                    push!(modified_nodes, u)
                    visited[u] = true
                end
            end

            rootnow = Rootindex[u]
            push!(a, i)
            push!(b, s[rootnow])

            u = i
            while Rootindex[u] == 0
                InForest[u] = true
                Rootindex[u] = rootnow

                if !visited[u]
                    push!(modified_nodes, u)
                    visited[u] = true
                end

                u = Next[u]
            end
        end
    end

    b ./= tau
    return sparsevec(a, b, G.n)
end

 
function sqrtzzpre_tau_parallel(G, tau, pp, s,alist)
    d = [length(G.nbr[j]) for j in 1:G.n]            

    a_threads = [Int[] for _ in 1:nthreads()]      
    b_threads = [Float64[] for _ in 1:nthreads()]     
    cnt = zeros(Int, nthreads())                    
    InForest_pool = [falses(G.n) for _ in 1:nthreads()]
    Next_pool = [fill(-1, G.n) for _ in 1:nthreads()]
    Rootindex_pool = [zeros(Int, G.n) for _ in 1:nthreads()]

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
            cntnow += 1;
            while !InForest[u]
                seed = rand()
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true
                    Rootindex[u] = u
                else
                    k = floor(Int, seed * (1 + d[u]))
                    Next[u] = G.nbr[u][k]
                    u = Next[u]
                    cntnow += 1;
                end
            end

            rootnow = Rootindex[u]
            push!(a_threads[thread_id], i)
            push!(b_threads[thread_id], s[rootnow])

            u = i
            while Rootindex[u] == 0
                InForest[u] = true
                Rootindex[u] = rootnow
                u = Next[u]
            end
        end
        cnt[thread_id] += cntnow
    end

    a = vcat(a_threads...)
    b = vcat(b_threads...)
    println("Total nodes processed: ", sum(cnt)/length(alist)/tau)
    return sparsevec(a, b ./ tau, G.n)  
end



function lazyz_parallel(G, l, r, s, d, pp,alist)
    a_threads = [Int[] for _ in 1:nthreads()]      
    b_threads = [Float64[] for _ in 1:nthreads()]    

    Threads.@threads for idx in eachindex(alist)
        i = alist[idx]
        thread_id = threadid()                      
        c = lazy(G, i, l, r, s, d)

        push!(a_threads[thread_id], i)
        push!(b_threads[thread_id], c)
    end

    a = vcat(a_threads...)
    b = vcat(b_threads...)

    return sparsevec(a, b, G.n)
end
