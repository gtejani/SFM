function [best_img,pair_img] = next_pair(match_matrix,old)
f = zeros(1,length(old));
max_f = 0;
for j = 1 : length(match_matrix)
    if ~ismember(j,old)
        for i = 1: length(old)
            f(i) = match_matrix(j,old(i));
        end
        f_sum = sum(f);
        if max_f < f_sum
            best_img = j;
            [~,ind]= max(f);
            pair_img = old(ind);
        end
        max_f = max(max_f,f_sum);
    end
end
end


