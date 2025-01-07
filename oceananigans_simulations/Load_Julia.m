function varargout = Load_Julia(filename,fields)
% Loads data from NETCDF file for Julia Oceananigans output
%
% inputs:
% - fielname: name of NETCDF data file (String)
% - fields: name(s) of fields to loads (Cell array), e.g. {'u','v'}
%
% outputs:
% - arrays: separate outputs, one array for each element of 'fields'

N = length(fields);

for n = 1:N
    varargout{n} = ncread(filename,fields{n});
end

end