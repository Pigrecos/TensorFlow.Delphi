unit hdf5dll;

// Delphi wrapper for HDF5 library(var hdf5dll-1.10.0-patch1)

// Auto-generated 2016-05-31 by hdf5pas.py.

interface

uses
  windows;

{$ALIGN ON}
{$MINENUMSIZE 4}

type
  int32_t = Integer;
  Pint32_t = ^int32_t;
  uint32_t = Cardinal;
  Puint32_t = ^uint32_t;
  int64_t = Int64;
  Pint64_t = ^int64_t;
  uint64_t = UInt64;
  Puint64_t = ^uint64_t;
  time_t = NativeInt;
  Ptime_t = ^time_t;
  size_t = NativeUInt;
  Psize_t = ^size_t;
  ssize_t = NativeInt;
  Pssize_t = ^ssize_t;
  off_t = NativeInt;
  Poff_t = ^off_t;
  PFILE = Pointer;

type
  hsize_t = UInt64;
  Phsize_t = ^hsize_t;
  hssize_t = Int64;
  Phssize_t = ^hssize_t;
  haddr_t = UInt64;
  Phaddr_t = ^haddr_t;

const
  HADDR_UNDEF = haddr_t(-1);

(* Version numbers *)
const
  H5_VERS_MAJOR = 1;  (* For major interface/format changes *)
  H5_VERS_MINOR = 10;  (* For minor interface/format changes *)
  H5_VERS_RELEASE = 0;  (* For tweaks, bug-fixes, or development *)
  H5_VERS_SUBRELEASE = 'patch1';  (* For pre-releases like snap0 *)
(* Empty string for real releases.           *)
  H5_VERS_INFO = 'HDF5 library version: 1.10.0-patch1';  (* Full version string *)

(*
 * Status return values.  Failed integer functions in HDF5 result almost
 * always in a negative value (unsigned failing functions sometimes return
 * zero for failure) while successfull return is non-negative (often zero).
 * The negative failure value is most commonly -1, but don't bet on it.  The
 * proper way to detect failure is something like:
 *
 *      if((dset = H5Dopen2(file, name)) < 0)
 *          fprintf(stderr, "unable to open the requested dataset\n");
 *)
type
  herr_t = Integer;
  Pherr_t = ^herr_t;

(*
 * Boolean type.  Successful return values are zero (false) or positive
 * (true). The typical true value is 1 but don't bet on it.  Boolean
 * functions cannot fail.  Functions that return `htri_t' however return zero
 * (false), positive (true), or negative (failure). The proper way to test
 * for truth from a htri_t function is:
 *
 *      if ((retval = H5Tcommitted(type))>0) {
 *          printf("data type is committed\n");
 *      } else if (!retval) {
 *          printf("data type is not committed\n");
 *      } else {
 *          printf("error determining whether data type is committed\n");
 *      }
 *)
type
  hbool_t = Boolean;
  Phbool_t = ^hbool_t;
  htri_t = Integer;
  Phtri_t = ^htri_t;

(*
 * The sizes of file objects have their own types defined here, use a 64-bit
 * type.
 *)
const
  HSIZE_UNDEF = hsize_t(hssize_t(-1));

(*
 * File addresses have their own types.
 *)
const
  HADDR_MAX = HADDR_UNDEF-1;

(* Common iteration orders *)
type
  PH5_iter_order_t = ^H5_iter_order_t;
  H5_iter_order_t =
    (H5_ITER_UNKNOWN = -1,  (* Unknown order *)
     H5_ITER_INC,  (* Increasing order *)
     H5_ITER_DEC,  (* Decreasing order *)
     H5_ITER_NATIVE,  (* No particular order, whatever is fastest *)
     H5_ITER_N);  (* Number of iteration orders *)

(* Iteration callback values *)
(* (Actually, any postive value will cause the iterator to stop and pass back
 *      that positive value to the function that called the iterator)
 *)
const
  H5_ITER_ERROR = -1;
  H5_ITER_CONT = 0;
  H5_ITER_STOP = 1;

(*
 * The types of indices on links in groups/attributes on objects.
 * Primarily used for "<do> <foo> by index" routines and for iterating over
 * links in groups/attributes on objects.
 *)
type
  PH5_index_t = ^H5_index_t;
  H5_index_t =
    (H5_INDEX_UNKNOWN = -1,  (* Unknown index type *)
     H5_INDEX_NAME,  (* Index on names *)
     H5_INDEX_CRT_ORDER,  (* Index on creation order *)
     H5_INDEX_N);  (* Number of indices defined *)

(*
 * Storage info struct used by H5O_info_t and H5F_info_t
 *)
type
  PH5_ih_info_t = ^H5_ih_info_t;
  PPH5_ih_info_t = ^PH5_ih_info_t;
  H5_ih_info_t = record
    index_size: hsize_t;  (* btree and/or list *)
    heap_size: hsize_t;
  end;

(*
 * Library type values.  Start with `1' instead of `0' because it makes the
 * tracing output look better when hid_t values are large numbers.  Change the
 * TYPE_BITS in H5I.c if the MAXID gets larger than 32 (an assertion will
 * fail otherwise).
 *
 * When adding types here, add a section to the 'misc19' test in test/tmisc.c
 * to verify that the H5I{inc|dec|get}_ref() routines work correctly with in.
 *
 *)
type
  PH5I_type_t = ^H5I_type_t;
  H5I_type_t =
    (H5I_UNINIT = -2,  (* uninitialized type *)
     H5I_BADID = -1,  (* invalid Type *)
     H5I_FILE = 1,  (* type ID for File objects *)
     H5I_GROUP,  (* type ID for Group objects *)
     H5I_DATATYPE,  (* type ID for Datatype objects *)
     H5I_DATASPACE,  (* type ID for Dataspace objects *)
     H5I_DATASET,  (* type ID for Dataset objects *)
     H5I_ATTR,  (* type ID for Attribute objects *)
     H5I_REFERENCE,  (* type ID for Reference objects *)
     H5I_VFL,  (* type ID for virtual file layer *)
     H5I_GENPROP_CLS,  (* type ID for generic property list classes *)
     H5I_GENPROP_LST,  (* type ID for generic property lists *)
     H5I_ERROR_CLASS,  (* type ID for error classes *)
     H5I_ERROR_MSG,  (* type ID for error messages *)
     H5I_ERROR_STACK,  (* type ID for error stacks *)
     H5I_NTYPES);  (* number of library types, MUST BE LAST! *)

(* Type of atoms to return to users *)
type
  hid_t = int64_t;
  Phid_t = ^hid_t;

(* An invalid object ID. This is also negative for error return. *)
const
  H5I_INVALID_HID = -1;

(*
 * Function for freeing objects. This function will be called with an object
 * ID type number and a pointer to the object. The function should free the
 * object and return non-negative to indicate that the object
 * can be removed from the ID type. If the function returns negative
 * (failure) then the object will remain in the ID type.
 *)
type
  H5I_free_t = function(p: Pointer): herr_t; cdecl;
  PH5I_free_t = ^H5I_free_t;

(* Type of the function to compare objects & keys *)
type
  H5I_search_func_t = function(obj: Pointer; id: hid_t; key: Pointer): Integer; cdecl;
  PH5I_search_func_t = ^H5I_search_func_t;
type
  PH5C_cache_incr_mode = ^H5C_cache_incr_mode;
  H5C_cache_incr_mode =
    (H5C_incr__off,
     H5C_incr__threshold);
type
  PH5C_cache_flash_incr_mode = ^H5C_cache_flash_incr_mode;
  H5C_cache_flash_incr_mode =
    (H5C_flash_incr__off,
     H5C_flash_incr__add_space);
type
  PH5C_cache_decr_mode = ^H5C_cache_decr_mode;
  H5C_cache_decr_mode =
    (H5C_decr__off,
     H5C_decr__threshold,
     H5C_decr__age_out,
     H5C_decr__age_out_with_threshold);

(* These typedefs are currently used for VL datatype allocation/freeing *)
type
  H5MM_allocate_t = function(size: size_t; alloc_info: Pointer): Pointer; cdecl;
  PH5MM_allocate_t = ^H5MM_allocate_t;
  H5MM_free_t = procedure(mem: Pointer; free_info: Pointer); cdecl;
  PH5MM_free_t = ^H5MM_free_t;

(*
 * Filter identifiers.  Values 0 through 255 are for filters defined by the
 * HDF5 library.  Values 256 through 511 are available for testing new
 * filters. Subsequent values should be obtained from the HDF5 development
 * team at hdf5dev@ncsa.uiuc.edu.  These values will never change because they
 * appear in the HDF5 files.
 *)
type
  H5Z_filter_t = Integer;
  PH5Z_filter_t = ^H5Z_filter_t;

(* Filter IDs *)
const
  H5Z_FILTER_ERROR = -1;  (* no filter *)
  H5Z_FILTER_NONE = 0;  (* reserved indefinitely *)
  H5Z_FILTER_DEFLATE = 1;  (* deflation like gzip *)
  H5Z_FILTER_SHUFFLE = 2;  (* shuffle the data *)
  H5Z_FILTER_FLETCHER32 = 3;  (* fletcher32 checksum of EDC *)
  H5Z_FILTER_SZIP = 4;  (* szip compression *)
  H5Z_FILTER_NBIT = 5;  (* nbit compression *)
  H5Z_FILTER_SCALEOFFSET = 6;  (* scale+offset compression *)
  H5Z_FILTER_RESERVED = 256;  (* filter ids below this value are reserved for library use *)
const
  H5Z_FILTER_MAX = 65535;  (* maximum filter id *)

(* General macros *)
const
  H5Z_FILTER_ALL = 0;  (* Symbol to remove all filters in H5Premove_filter *)
  H5Z_MAX_NFILTERS = 32;  (* Maximum number of filters allowed in a pipeline *)
(* (should probably be allowed to be an
 * unlimited amount, but currently each
 * filter uses a bit in a 32-bit field,
 * so the format would have to be
 * changed to accomodate that)
 *)

(* Flags for filter definition (stored) *)
const
  H5Z_FLAG_DEFMASK = 255;  (* definition flag mask *)
  H5Z_FLAG_MANDATORY = 0;  (* filter is mandatory *)
  H5Z_FLAG_OPTIONAL = 1;  (* filter is optional *)

(* Additional flags for filter invocation (not stored) *)
const
  H5Z_FLAG_INVMASK = 65280;  (* invocation flag mask *)
  H5Z_FLAG_REVERSE = 256;  (* reverse direction; read *)
  H5Z_FLAG_SKIP_EDC = 512;  (* skip EDC filters for read *)

(* Special parameters for szip compression *)
(* [These are aliases for the similar definitions in szlib.h, which we can't
 * include directly due to the duplication of various symbols with the zlib.h
 * header file] *)
const
  H5_SZIP_ALLOW_K13_OPTION_MASK = 1;
  H5_SZIP_CHIP_OPTION_MASK = 2;
  H5_SZIP_EC_OPTION_MASK = 4;
  H5_SZIP_NN_OPTION_MASK = 32;
  H5_SZIP_MAX_PIXELS_PER_BLOCK = 32;

(* Macros for the shuffle filter *)
const
  H5Z_SHUFFLE_USER_NPARMS = 0;  (* Number of parameters that users can set *)
  H5Z_SHUFFLE_TOTAL_NPARMS = 1;  (* Total number of parameters for filter *)

(* Macros for the szip filter *)
const
  H5Z_SZIP_USER_NPARMS = 2;  (* Number of parameters that users can set *)
  H5Z_SZIP_TOTAL_NPARMS = 4;  (* Total number of parameters for filter *)
  H5Z_SZIP_PARM_MASK = 0;  (* "User" parameter for option mask *)
  H5Z_SZIP_PARM_PPB = 1;  (* "User" parameter for pixels-per-block *)
  H5Z_SZIP_PARM_BPP = 2;  (* "Local" parameter for bits-per-pixel *)
  H5Z_SZIP_PARM_PPS = 3;  (* "Local" parameter for pixels-per-scanline *)

(* Macros for the nbit filter *)
const
  H5Z_NBIT_USER_NPARMS = 0;  (* Number of parameters that users can set *)

(* Macros for the scale offset filter *)
const
  H5Z_SCALEOFFSET_USER_NPARMS = 2;  (* Number of parameters that users can set *)

(* Special parameters for ScaleOffset filter*)
const
  H5Z_SO_INT_MINBITS_DEFAULT = 0;
type
  PH5Z_SO_scale_type_t = ^H5Z_SO_scale_type_t;
  H5Z_SO_scale_type_t =
    (H5Z_SO_FLOAT_DSCALE,
     H5Z_SO_FLOAT_ESCALE = 1,
     H5Z_SO_INT = 2);

(* Current version of the H5Z_class_t struct *)
const
  H5Z_CLASS_T_VERS = 1;

(* Values to decide if EDC is enabled for reading data *)
type
  PH5Z_EDC_t = ^H5Z_EDC_t;
  H5Z_EDC_t =
    (H5Z_ERROR_EDC = -1,  (* error value *)
     H5Z_DISABLE_EDC,
     H5Z_ENABLE_EDC = 1,
     H5Z_NO_EDC = 2);  (* must be the last *)

(* Bit flags for H5Zget_filter_info *)
const
  H5Z_FILTER_CONFIG_ENCODE_ENABLED = 1;
  H5Z_FILTER_CONFIG_DECODE_ENABLED = 2;

(* Return values for filter callback function *)
type
  PH5Z_cb_return_t = ^H5Z_cb_return_t;
  H5Z_cb_return_t =
    (H5Z_CB_ERROR = -1,
     H5Z_CB_FAIL,  (* I/O should fail if filter fails. *)
     H5Z_CB_CONT = 1,  (* I/O continues if filter fails. *)
     H5Z_CB_NO = 2);

(* Filter callback function definition *)
type
  H5Z_filter_func_t = function(filter: H5Z_filter_t; buf: Pointer; buf_size: size_t; op_data: Pointer): H5Z_cb_return_t; cdecl;
  PH5Z_filter_func_t = ^H5Z_filter_func_t;

(* Structure for filter callback property *)
type
  PH5Z_cb_t = ^H5Z_cb_t;
  PPH5Z_cb_t = ^PH5Z_cb_t;
  H5Z_cb_t = record
    func: H5Z_filter_func_t;
    op_data: Pointer;
  end;

(*
 * Before a dataset gets created, the "can_apply" callbacks for any filters used
 * in the dataset creation property list are called
 * with the dataset's dataset creation property list, the dataset's datatype and
 * a dataspace describing a chunk (for chunked dataset storage).
 *
 * The "can_apply" callback must determine if the combination of the dataset
 * creation property list setting, the datatype and the dataspace represent a
 * valid combination to apply this filter to.  For example, some cases of
 * invalid combinations may involve the filter not operating correctly on
 * certain datatypes (or certain datatype sizes), or certain sizes of the chunk
 * dataspace.
 *
 * The "can_apply" callback can be the NULL pointer, in which case, the library
 * will assume that it can apply to any combination of dataset creation
 * property list values, datatypes and dataspaces.
 *
 * The "can_apply" callback returns positive a valid combination, zero for an
 * invalid combination and negative for an error.
 *)
type
  H5Z_can_apply_func_t = function(dcpl_id: hid_t; type_id: hid_t; space_id: hid_t): htri_t; cdecl;
  PH5Z_can_apply_func_t = ^H5Z_can_apply_func_t;

(*
 * After the "can_apply" callbacks are checked for new datasets, the "set_local"
 * callbacks for any filters used in the dataset creation property list are
 * called.  These callbacks receive the dataset's private copy of the dataset
 * creation property list passed in to H5Dcreate (i.e. not the actual property
 * list passed in to H5Dcreate) and the datatype ID passed in to H5Dcreate
 * (which is not copied and should not be modified) and a dataspace describing
 * the chunk (for chunked dataset storage) (which should also not be modified).
 *
 * The "set_local" callback must set any parameters that are specific to this
 * dataset, based on the combination of the dataset creation property list
 * values, the datatype and the dataspace.  For example, some filters perform
 * different actions based on different datatypes (or datatype sizes) or
 * different number of dimensions or dataspace sizes.
 *
 * The "set_local" callback can be the NULL pointer, in which case, the library
 * will assume that there are no dataset-specific settings for this filter.
 *
 * The "set_local" callback must return non-negative on success and negative
 * for an error.
 *)
type
  H5Z_set_local_func_t = function(dcpl_id: hid_t; type_id: hid_t; space_id: hid_t): herr_t; cdecl;
  PH5Z_set_local_func_t = ^H5Z_set_local_func_t;

(*
 * A filter gets definition flags and invocation flags (defined above), the
 * client data array and size defined when the filter was added to the
 * pipeline, the size in bytes of the data on which to operate, and pointers
 * to a buffer and its allocated size.
 *
 * The filter should store the result in the supplied buffer if possible,
 * otherwise it can allocate a new buffer, freeing the original.  The
 * allocated size of the new buffer should be returned through the BUF_SIZE
 * pointer and the new buffer through the BUF pointer.
 *
 * The return value from the filter is the number of bytes in the output
 * buffer. If an error occurs then the function should return zero and leave
 * all pointer arguments unchanged.
 *)
type
  H5Z_func_t = function(flags: Cardinal; cd_nelmts: size_t; cd_values: PCardinal; nbytes: size_t; buf_size: Psize_t; buf: PPointer): size_t; cdecl;
  PH5Z_func_t = ^H5Z_func_t;

(*
 * The filter table maps filter identification numbers to structs that
 * contain a pointers to the filter function and timing statistics.
 *)
type
  PH5Z_class2_t = ^H5Z_class2_t;
  PPH5Z_class2_t = ^PH5Z_class2_t;
  H5Z_class2_t = record
    version: Integer;  (* Version number of the H5Z_class_t struct *)
    id: H5Z_filter_t;  (* Filter ID number *)
    encoder_present: Cardinal;  (* Does this filter have an encoder? *)
    decoder_present: Cardinal;  (* Does this filter have a decoder? *)
    name: PAnsiChar;  (* Comment for debugging *)
    can_apply: H5Z_can_apply_func_t;  (* The "can apply" callback for a filter *)
    set_local: H5Z_set_local_func_t;  (* The "set local" callback for a filter *)
    filter: H5Z_func_t;  (* The actual filter function *)
  end;

(* Plugin type used by the plugin library *)
type
  PH5PL_type_t = ^H5PL_type_t;
  H5PL_type_t =
    (H5PL_TYPE_ERROR = -1,  (* error *)
     H5PL_TYPE_FILTER,  (* filter *)
     H5PL_TYPE_NONE = 1);  (* this must be last! *)

(* Common dynamic plugin type flags used by the set/get_loading_state functions *)
const
  H5PL_FILTER_PLUGIN = 1;
  H5PL_ALL_PLUGIN = 65535;

(* These are the various classes of datatypes *)
(* If this goes over 16 types (0-15), the file format will need to change) *)
type
  PH5T_class_t = ^H5T_class_t;
  H5T_class_t =
    (H5T_NO_CLASS = -1,  (* error *)
     H5T_INTEGER,  (* integer types *)
     H5T_FLOAT = 1,  (* floating-point types *)
     H5T_TIME = 2,  (* date and time types *)
     H5T_STRING = 3,  (* character string types *)
     H5T_BITFIELD = 4,  (* bit field types *)
     H5T_OPAQUE = 5,  (* opaque types *)
     H5T_COMPOUND = 6,  (* compound types *)
     H5T_REFERENCE = 7,  (* reference types *)
     H5T_ENUM = 8,  (* enumeration types *)
     H5T_VLEN = 9,  (* Variable-Length types *)
     H5T_ARRAY = 10,  (* Array types *)

     H5T_NCLASSES);  (* this must be last *)

(* Byte orders *)
type
  PH5T_order_t = ^H5T_order_t;
  H5T_order_t =
    (H5T_ORDER_ERROR = -1,  (* error *)
     H5T_ORDER_LE,  (* little endian *)
     H5T_ORDER_BE = 1,  (* bit endian *)
     H5T_ORDER_VAX = 2,  (* VAX mixed endian *)
     H5T_ORDER_MIXED = 3,  (* Compound type with mixed member orders *)
     H5T_ORDER_NONE = 4);  (* no particular order (strings, bits,..) *)
(*H5T_ORDER_NONE must be last *)

(* Types of integer sign schemes *)
type
  PH5T_sign_t = ^H5T_sign_t;
  H5T_sign_t =
    (H5T_SGN_ERROR = -1,  (* error *)
     H5T_SGN_NONE,  (* this is an unsigned type *)
     H5T_SGN_2 = 1,  (* two's complement *)

     H5T_NSGN = 2);  (* this must be last! *)

(* Floating-point normalization schemes *)
type
  PH5T_norm_t = ^H5T_norm_t;
  H5T_norm_t =
    (H5T_NORM_ERROR = -1,  (* error *)
     H5T_NORM_IMPLIED,  (* msb of mantissa isn't stored, always 1 *)
     H5T_NORM_MSBSET = 1,  (* msb of mantissa is always 1 *)
     H5T_NORM_NONE = 2);  (* not normalized *)
(*H5T_NORM_NONE must be last *)

(*
 * Character set to use for text strings.  Do not change these values since
 * they appear in HDF5 files!
 *)
type
  PH5T_cset_t = ^H5T_cset_t;
  H5T_cset_t =
    (H5T_CSET_ERROR = -1,  (* error *)
     H5T_CSET_ASCII,  (* US ASCII *)
     H5T_CSET_UTF8 = 1,  (* UTF-8 Unicode encoding *)
     H5T_CSET_RESERVED_2 = 2,  (* reserved for later use *)
     H5T_CSET_RESERVED_3 = 3,  (* reserved for later use *)
     H5T_CSET_RESERVED_4 = 4,  (* reserved for later use *)
     H5T_CSET_RESERVED_5 = 5,  (* reserved for later use *)
     H5T_CSET_RESERVED_6 = 6,  (* reserved for later use *)
     H5T_CSET_RESERVED_7 = 7,  (* reserved for later use *)
     H5T_CSET_RESERVED_8 = 8,  (* reserved for later use *)
     H5T_CSET_RESERVED_9 = 9,  (* reserved for later use *)
     H5T_CSET_RESERVED_10 = 10,  (* reserved for later use *)
     H5T_CSET_RESERVED_11 = 11,  (* reserved for later use *)
     H5T_CSET_RESERVED_12 = 12,  (* reserved for later use *)
     H5T_CSET_RESERVED_13 = 13,  (* reserved for later use *)
     H5T_CSET_RESERVED_14 = 14,  (* reserved for later use *)
     H5T_CSET_RESERVED_15 = 15);  (* reserved for later use *)
const
  H5T_NCSET = H5T_CSET_RESERVED_2;  (* Number of character sets actually defined *)

(*
 * Type of padding to use in character strings.  Do not change these values
 * since they appear in HDF5 files!
 *)
type
  PH5T_str_t = ^H5T_str_t;
  H5T_str_t =
    (H5T_STR_ERROR = -1,  (* error *)
     H5T_STR_NULLTERM,  (* null terminate like in C *)
     H5T_STR_NULLPAD = 1,  (* pad with nulls *)
     H5T_STR_SPACEPAD = 2,  (* pad with spaces like in Fortran *)
     H5T_STR_RESERVED_3 = 3,  (* reserved for later use *)
     H5T_STR_RESERVED_4 = 4,  (* reserved for later use *)
     H5T_STR_RESERVED_5 = 5,  (* reserved for later use *)
     H5T_STR_RESERVED_6 = 6,  (* reserved for later use *)
     H5T_STR_RESERVED_7 = 7,  (* reserved for later use *)
     H5T_STR_RESERVED_8 = 8,  (* reserved for later use *)
     H5T_STR_RESERVED_9 = 9,  (* reserved for later use *)
     H5T_STR_RESERVED_10 = 10,  (* reserved for later use *)
     H5T_STR_RESERVED_11 = 11,  (* reserved for later use *)
     H5T_STR_RESERVED_12 = 12,  (* reserved for later use *)
     H5T_STR_RESERVED_13 = 13,  (* reserved for later use *)
     H5T_STR_RESERVED_14 = 14,  (* reserved for later use *)
     H5T_STR_RESERVED_15 = 15);  (* reserved for later use *)
const
  H5T_NSTR = H5T_STR_RESERVED_3;  (* num H5T_str_t types actually defined *)

(* Type of padding to use in other atomic types *)
type
  PH5T_pad_t = ^H5T_pad_t;
  H5T_pad_t =
    (H5T_PAD_ERROR = -1,  (* error *)
     H5T_PAD_ZERO,  (* always set to zero *)
     H5T_PAD_ONE = 1,  (* always set to one *)
     H5T_PAD_BACKGROUND = 2,  (* set to background value *)

     H5T_NPAD = 3);  (* THIS MUST BE LAST *)

(* Commands sent to conversion functions *)
type
  PH5T_cmd_t = ^H5T_cmd_t;
  H5T_cmd_t =
    (H5T_CONV_INIT,  (* query and/or initialize private data *)
     H5T_CONV_CONV = 1,  (* convert data from source to dest datatype *)
     H5T_CONV_FREE = 2);  (* function is being removed from path *)

(* How is the `bkg' buffer used by the conversion function? *)
type
  PH5T_bkg_t = ^H5T_bkg_t;
  H5T_bkg_t =
    (H5T_BKG_NO,  (* background buffer is not needed, send NULL *)
     H5T_BKG_TEMP = 1,  (* bkg buffer used as temp storage only *)
     H5T_BKG_YES = 2);  (* init bkg buf with data before conversion *)

(* Type conversion client data *)
type
  PH5T_cdata_t = ^H5T_cdata_t;
  PPH5T_cdata_t = ^PH5T_cdata_t;
  H5T_cdata_t = record
    command: H5T_cmd_t;  (* what should the conversion function do? *)
    need_bkg: H5T_bkg_t;  (* is the background buffer needed? *)
    recalc: hbool_t;  (* recalculate private data *)
    priv: Pointer;  (* private data *)
  end;

(* Conversion function persistence *)
type
  PH5T_pers_t = ^H5T_pers_t;
  H5T_pers_t =
    (H5T_PERS_DONTCARE = -1,  (* wild card *)
     H5T_PERS_HARD,  (* hard conversion function *)
     H5T_PERS_SOFT = 1);  (* soft conversion function *)

(* The order to retrieve atomic native datatype *)
type
  PH5T_direction_t = ^H5T_direction_t;
  H5T_direction_t =
    (H5T_DIR_DEFAULT,  (* default direction is inscendent *)
     H5T_DIR_ASCEND = 1,  (* in inscendent order *)
     H5T_DIR_DESCEND = 2);  (* in descendent order *)

(* The exception type passed into the conversion callback function *)
type
  PH5T_conv_except_t = ^H5T_conv_except_t;
  H5T_conv_except_t =
    (H5T_CONV_EXCEPT_RANGE_HI,  (* source value is greater than destination's range *)
     H5T_CONV_EXCEPT_RANGE_LOW = 1,  (* source value is less than destination's range *)
     H5T_CONV_EXCEPT_PRECISION = 2,  (* source value loses precision in destination *)
     H5T_CONV_EXCEPT_TRUNCATE = 3,  (* source value is truncated in destination *)
     H5T_CONV_EXCEPT_PINF = 4,  (* source value is positive infinity(floating number) *)
     H5T_CONV_EXCEPT_NINF = 5,  (* source value is negative infinity(floating number) *)
     H5T_CONV_EXCEPT_NAN = 6);  (* source value is NaN(floating number) *)

(* The return value from conversion callback function H5T_conv_except_func_t *)
type
  PH5T_conv_ret_t = ^H5T_conv_ret_t;
  H5T_conv_ret_t =
    (H5T_CONV_ABORT = -1,  (* abort conversion *)
     H5T_CONV_UNHANDLED,  (* callback function failed to handle the exception *)
     H5T_CONV_HANDLED = 1);  (* callback function handled the exception successfully *)

(* Variable Length Datatype struct in memory *)
(* (This is only used for VL sequences, not VL strings, which are stored in char *'s) *)
type
  Phvl_t = ^hvl_t;
  PPhvl_t = ^Phvl_t;
  hvl_t = record
    len: size_t;  (* Length of VL data (in base type units) *)
    p: Pointer;  (* Pointer to VL data *)
  end;

(* Variable Length String information *)
const
  H5T_VARIABLE = size_t(-1);  (* Indicate that a string is variable length (null-terminated in C, instead of fixed length) *)

(* Opaque information *)
const
  H5T_OPAQUE_TAG_MAX = 256;  (* Maximum length of an opaque tag *)
(* This could be raised without too much difficulty *)

(* All datatype conversion functions are... *)
type
  H5T_conv_t = function(src_id: hid_t; dst_id: hid_t; cdata: PH5T_cdata_t; nelmts: size_t; buf_stride: size_t; bkg_stride: size_t; buf: Pointer; bkg: Pointer; dset_xfer_plist: hid_t): herr_t; cdecl;
  PH5T_conv_t = ^H5T_conv_t;

(* Exception handler.  If an exception like overflow happenes during conversion,
 * this function is called if it's registered through H5Pset_type_conv_cb.
 *)
type
  H5T_conv_except_func_t = function(except_type: H5T_conv_except_t; src_id: hid_t; dst_id: hid_t; src_buf: Pointer; dst_buf: Pointer; user_data: Pointer): H5T_conv_ret_t; cdecl;
  PH5T_conv_except_func_t = ^H5T_conv_except_func_t;
const
  H5AC__CURR_CACHE_CONFIG_VERSION = 1;
  H5AC__MAX_TRACE_FILE_NAME_LEN = 1024;
const
  H5AC_METADATA_WRITE_STRATEGY__PROCESS_0_ONLY = 0;
  H5AC_METADATA_WRITE_STRATEGY__DISTRIBUTED = 1;
type
  PH5AC_cache_config_t = ^H5AC_cache_config_t;
  PPH5AC_cache_config_t = ^PH5AC_cache_config_t;
  H5AC_cache_config_t = record
    version: Integer;
    rpt_fcn_enabled: hbool_t;
    open_trace_file: hbool_t;
    close_trace_file: hbool_t;
    trace_file_name: array[0..H5AC__MAX_TRACE_FILE_NAME_LEN] of AnsiChar;
    evictions_enabled: hbool_t;
    set_initial_size: hbool_t;
    initial_size: size_t;
    min_clean_fraction: Double;
    max_size: size_t;
    min_size: size_t;
    epoch_length: Integer;
    incr_mode: H5C_cache_incr_mode;
    lower_hr_threshold: Double;
    increment: Double;
    apply_max_increment: hbool_t;
    max_increment: size_t;
    flash_incr_mode: H5C_cache_flash_incr_mode;
    flash_multiple: Double;
    flash_threshold: Double;
    decr_mode: H5C_cache_decr_mode;
    upper_hr_threshold: Double;
    decrement: Double;
    apply_max_decrement: hbool_t;
    max_decrement: size_t;
    epochs_before_eviction: Integer;
    apply_empty_reserve: hbool_t;
    empty_reserve: Double;
    dirty_bytes_threshold: size_t;
    metadata_write_strategy: Integer;
  end;

(* Macros used to "unset" chunk cache configuration parameters *)
const
  H5D_CHUNK_CACHE_NSLOTS_DEFAULT = size_t(-1);
  H5D_CHUNK_CACHE_NBYTES_DEFAULT = size_t(-1);
  H5D_CHUNK_CACHE_W0_DEFAULT = -1.0;

(* Bit flags for the H5Pset_chunk_opts() and H5Pget_chunk_opts() *)
const
  H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS = 2;

(* Property names for H5LTDdirect_chunk_write *)
const
  H5D_XFER_DIRECT_CHUNK_WRITE_FLAG_NAME = 'direct_chunk_flag';
  H5D_XFER_DIRECT_CHUNK_WRITE_FILTERS_NAME = 'direct_chunk_filters';
  H5D_XFER_DIRECT_CHUNK_WRITE_OFFSET_NAME = 'direct_chunk_offset';
  H5D_XFER_DIRECT_CHUNK_WRITE_DATASIZE_NAME = 'direct_chunk_datasize';

(* Values for the H5D_LAYOUT property *)
type
  PH5D_layout_t = ^H5D_layout_t;
  H5D_layout_t =
    (H5D_LAYOUT_ERROR = -1,

     H5D_COMPACT,  (* raw data is very small *)
     H5D_CONTIGUOUS = 1,  (* the default *)
     H5D_CHUNKED = 2,  (* slow and fancy *)
     H5D_VIRTUAL = 3,  (* actual data is stored in other datasets *)
     H5D_NLAYOUTS = 4);  (* this one must be last! *)

(* Types of chunk index data structures *)
type
  PH5D_chunk_index_t = ^H5D_chunk_index_t;
  H5D_chunk_index_t =
    (H5D_CHUNK_IDX_BTREE,  (* v1 B-tree index (default) *)
     H5D_CHUNK_IDX_SINGLE = 1,  (* Single Chunk index (cur dims[]=max dims[]=chunk dims[]; filtered & non-filtered) *)
     H5D_CHUNK_IDX_NONE = 2,  (* Implicit: No Index (H5D_ALLOC_TIME_EARLY, non-filtered, fixed dims) *)
     H5D_CHUNK_IDX_FARRAY = 3,  (* Fixed array (for 0 unlimited dims) *)
     H5D_CHUNK_IDX_EARRAY = 4,  (* Extensible array (for 1 unlimited dim) *)
     H5D_CHUNK_IDX_BT2 = 5,  (* v2 B-tree index (for >1 unlimited dims) *)
     H5D_CHUNK_IDX_NTYPES);  (* this one must be last! *)

(* Values for the space allocation time property *)
type
  PH5D_alloc_time_t = ^H5D_alloc_time_t;
  H5D_alloc_time_t =
    (H5D_ALLOC_TIME_ERROR = -1,
     H5D_ALLOC_TIME_DEFAULT,
     H5D_ALLOC_TIME_EARLY = 1,
     H5D_ALLOC_TIME_LATE = 2,
     H5D_ALLOC_TIME_INCR = 3);

(* Values for the status of space allocation *)
type
  PH5D_space_status_t = ^H5D_space_status_t;
  H5D_space_status_t =
    (H5D_SPACE_STATUS_ERROR = -1,
     H5D_SPACE_STATUS_NOT_ALLOCATED,
     H5D_SPACE_STATUS_PART_ALLOCATED = 1,
     H5D_SPACE_STATUS_ALLOCATED = 2);

(* Values for time of writing fill value property *)
type
  PH5D_fill_time_t = ^H5D_fill_time_t;
  H5D_fill_time_t =
    (H5D_FILL_TIME_ERROR = -1,
     H5D_FILL_TIME_ALLOC,
     H5D_FILL_TIME_NEVER = 1,
     H5D_FILL_TIME_IFSET = 2);

(* Values for fill value status *)
type
  PH5D_fill_value_t = ^H5D_fill_value_t;
  H5D_fill_value_t =
    (H5D_FILL_VALUE_ERROR = -1,
     H5D_FILL_VALUE_UNDEFINED,
     H5D_FILL_VALUE_DEFAULT = 1,
     H5D_FILL_VALUE_USER_DEFINED = 2);

(* Values for VDS bounds option *)
type
  PH5D_vds_view_t = ^H5D_vds_view_t;
  H5D_vds_view_t =
    (H5D_VDS_ERROR = -1,
     H5D_VDS_FIRST_MISSING,
     H5D_VDS_LAST_AVAILABLE = 1);

(* Callback for H5Pset_append_flush() in a dataset access property list *)
type
  H5D_append_cb_t = function(dataset_id: hid_t; cur_dims: Phsize_t; op_data: Pointer): herr_t; cdecl;
  PH5D_append_cb_t = ^H5D_append_cb_t;

(* Define the operator function pointer for H5Diterate() *)
type
  H5D_operator_t = function(elem: Pointer; type_id: hid_t; ndim: Cardinal; point: Phsize_t; operator_data: Pointer): herr_t; cdecl;
  PH5D_operator_t = ^H5D_operator_t;

(* Define the operator function pointer for H5Dscatter() *)
type
  H5D_scatter_func_t = function(src_buf: PPointer; src_buf_bytes_used: Psize_t; op_data: Pointer): herr_t; cdecl;
  PH5D_scatter_func_t = ^H5D_scatter_func_t;

(* Define the operator function pointer for H5Dgather() *)
type
  H5D_gather_func_t = function(dst_buf: Pointer; dst_buf_bytes_used: size_t; op_data: Pointer): herr_t; cdecl;
  PH5D_gather_func_t = ^H5D_gather_func_t;

(* Value for the default error stack *)
const
  H5E_DEFAULT = hid_t(0);

(* Different kinds of error information *)
type
  PH5E_type_t = ^H5E_type_t;
  H5E_type_t =
    (H5E_MAJOR,
     H5E_MINOR);

(* Information about an error; element of error stack *)
type
  PH5E_error2_t = ^H5E_error2_t;
  PPH5E_error2_t = ^PH5E_error2_t;
  H5E_error2_t = record
    cls_id: hid_t;  (* class ID *)
    maj_num: hid_t;  (* major error ID *)
    min_num: hid_t;  (* minor error number *)
    line: Cardinal;  (* line in file where error occurs *)
    func_name: PAnsiChar;  (* function in which error occurred *)
    file_name: PAnsiChar;  (* file in which error occurred *)
    desc: PAnsiChar;  (* optional supplied description *)
  end;

(* Error stack traversal direction *)
type
  PH5E_direction_t = ^H5E_direction_t;
  H5E_direction_t =
    (H5E_WALK_UPWARD,  (* begin deep, end at API function *)
     H5E_WALK_DOWNWARD = 1);  (* begin at API function, end deep *)

(* Error stack traversal callback function pointers *)
type
  H5E_walk2_t = function(n: Cardinal; err_desc: PH5E_error2_t; client_data: Pointer): herr_t; cdecl;
  PH5E_walk2_t = ^H5E_walk2_t;
  H5E_auto2_t = function(estack: hid_t; client_data: Pointer): herr_t; cdecl;
  PH5E_auto2_t = ^H5E_auto2_t;

(* Define atomic datatypes *)
const
  H5S_ALL = hid_t(0);
  H5S_UNLIMITED = HSIZE_UNDEF;

(* Define user-level maximum number of dimensions *)
const
  H5S_MAX_RANK = 32;

(* Different types of dataspaces *)
type
  PH5S_class_t = ^H5S_class_t;
  H5S_class_t =
    (H5S_NO_CLASS = -1,  (* error *)
     H5S_SCALAR,  (* scalar variable *)
     H5S_SIMPLE = 1,  (* simple data space *)
     H5S_NULL = 2);  (* null data space *)

(* Different ways of combining selections *)
type
  PH5S_seloper_t = ^H5S_seloper_t;
  H5S_seloper_t =
    (H5S_SELECT_NOOP = -1,  (* error *)
     H5S_SELECT_SET,  (* Select "set" operation *)
     H5S_SELECT_OR,
(* Binary "or" operation for hyperslabs
 * (add new selection to existing selection)
 * Original region:  AAAAAAAAAA
 * New region:             BBBBBBBBBB
 * A or B:           CCCCCCCCCCCCCCCC
 *)
     H5S_SELECT_AND,
(* Binary "and" operation for hyperslabs
 * (only leave overlapped regions in selection)
 * Original region:  AAAAAAAAAA
 * New region:             BBBBBBBBBB
 * A and B:                CCCC
 *)
     H5S_SELECT_XOR,
(* Binary "xor" operation for hyperslabs
 * (only leave non-overlapped regions in selection)
 * Original region:  AAAAAAAAAA
 * New region:             BBBBBBBBBB
 * A xor B:          CCCCCC    CCCCCC
 *)
     H5S_SELECT_NOTB,
(* Binary "not" operation for hyperslabs
 * (only leave non-overlapped regions in original selection)
 * Original region:  AAAAAAAAAA
 * New region:             BBBBBBBBBB
 * A not B:          CCCCCC
 *)
     H5S_SELECT_NOTA,
(* Binary "not" operation for hyperslabs
 * (only leave non-overlapped regions in new selection)
 * Original region:  AAAAAAAAAA
 * New region:             BBBBBBBBBB
 * B not A:                    CCCCCC
 *)
     H5S_SELECT_APPEND,  (* Append elements to end of point selection *)
     H5S_SELECT_PREPEND,  (* Prepend elements to beginning of point selection *)
     H5S_SELECT_INVALID);  (* Invalid upper bound on selection operations *)

(* Enumerated type for the type of selection *)
type
  PH5S_sel_type = ^H5S_sel_type;
  H5S_sel_type =
    (H5S_SEL_ERROR = -1,  (* Error *)
     H5S_SEL_NONE,  (* Nothing selected *)
     H5S_SEL_POINTS = 1,  (* Sequence of points selected *)
     H5S_SEL_HYPERSLABS = 2,  (* "New-style" hyperslab selection defined *)
     H5S_SEL_ALL = 3,  (* Entire extent selected *)
     H5S_SEL_N);  (* THIS MUST BE LAST *)

(* Maximum length of a link's name *)
(* (encoded in a 32-bit unsigned integer) *)
const
  H5L_MAX_LINK_NAME_LEN = uint32_t(-1);  (* (4GB - 1) *)

(* Macro to indicate operation occurs on same location *)
const
  H5L_SAME_LOC = hid_t(0);

(* Current version of the H5L_class_t struct *)
const
  H5L_LINK_CLASS_T_VERS = 0;

(* Link class types.
 * Values less than 64 are reserved for the HDF5 library's internal use.
 * Values 64 to 255 are for "user-defined" link class types; these types are
 * defined by HDF5 but their behavior can be overridden by users.
 * Users who want to create new classes of links should contact the HDF5
 * development team at hdfhelp@ncsa.uiuc.edu .
 * These values can never change because they appear in HDF5 files.
 *)
type
  PH5L_type_t = ^H5L_type_t;
  H5L_type_t =
    (H5L_TYPE_ERROR = -1,  (* Invalid link type id *)
     H5L_TYPE_HARD,  (* Hard link id *)
     H5L_TYPE_SOFT = 1,  (* Soft link id *)
     H5L_TYPE_EXTERNAL = 64,  (* External link id *)
     H5L_TYPE_MAX = 255);  (* Maximum link type id *)
const
  H5L_TYPE_BUILTIN_MAX = H5L_TYPE_SOFT;  (* Maximum value link value for "built-in" link types *)
  H5L_TYPE_UD_MIN = H5L_TYPE_EXTERNAL;  (* Link ids at or above this value are "user-defined" link types. *)

(* Information struct for link (for H5Lget_info/H5Lget_info_by_idx) *)
type
  PH5L_info_t = ^H5L_info_t;
  PPH5L_info_t = ^PH5L_info_t;
  H5L_info_t = record
    typ: H5L_type_t;  (* Type of link *)
    corder_valid: hbool_t;  (* Indicate if creation order is valid *)
    corder: int64_t;  (* Creation order *)
    cset: H5T_cset_t;  (* Character set of link name *)
    case Integer of
      1: (address: haddr_t);  (* Address hard link points to *)
      2: (val_size: size_t);  (* Size of a soft link or UD link value *)
  end;

(* The H5L_class_t struct can be used to override the behavior of a
 * "user-defined" link class. Users should populate the struct with callback
 * functions defined below.
 *)
(* Callback prototypes for user-defined links *)
(* Link creation callback *)
type
  H5L_create_func_t = function(link_name: PAnsiChar; loc_group: hid_t; lnkdata: Pointer; lnkdata_size: size_t; lcpl_id: hid_t): herr_t; cdecl;
  PH5L_create_func_t = ^H5L_create_func_t;

(* Callback for when the link is moved *)
type
  H5L_move_func_t = function(new_name: PAnsiChar; new_loc: hid_t; lnkdata: Pointer; lnkdata_size: size_t): herr_t; cdecl;
  PH5L_move_func_t = ^H5L_move_func_t;

(* Callback for when the link is copied *)
type
  H5L_copy_func_t = function(new_name: PAnsiChar; new_loc: hid_t; lnkdata: Pointer; lnkdata_size: size_t): herr_t; cdecl;
  PH5L_copy_func_t = ^H5L_copy_func_t;

(* Callback during link traversal *)
type
  H5L_traverse_func_t = function(link_name: PAnsiChar; cur_group: hid_t; lnkdata: Pointer; lnkdata_size: size_t; lapl_id: hid_t): hid_t; cdecl;
  PH5L_traverse_func_t = ^H5L_traverse_func_t;

(* Callback for when the link is deleted *)
type
  H5L_delete_func_t = function(link_name: PAnsiChar; file_: hid_t; lnkdata: Pointer; lnkdata_size: size_t): herr_t; cdecl;
  PH5L_delete_func_t = ^H5L_delete_func_t;

(* Callback for querying the link *)
(* Returns the size of the buffer needed *)
type
  H5L_query_func_t = function(link_name: PAnsiChar; lnkdata: Pointer; lnkdata_size: size_t; buf: Pointer; buf_size: size_t): ssize_t; cdecl;
  PH5L_query_func_t = ^H5L_query_func_t;

(* User-defined link types *)
type
  PH5L_class_t = ^H5L_class_t;
  PPH5L_class_t = ^PH5L_class_t;
  H5L_class_t = record
    version: Integer;  (* Version number of this struct *)
    id: H5L_type_t;  (* Link type ID *)
    comment: PAnsiChar;  (* Comment for debugging *)
    create_func: H5L_create_func_t;  (* Callback during link creation *)
    move_func: H5L_move_func_t;  (* Callback after moving link *)
    copy_func: H5L_copy_func_t;  (* Callback after copying link *)
    trav_func: H5L_traverse_func_t;  (* Callback during link traversal *)
    del_func: H5L_delete_func_t;  (* Callback for link deletion *)
    query_func: H5L_query_func_t;  (* Callback for queries *)
  end;

(* Prototype for H5Literate/H5Literate_by_name() operator *)
type
  H5L_iterate_t = function(group: hid_t; name: PAnsiChar; info: PH5L_info_t; op_data: Pointer): herr_t; cdecl;
  PH5L_iterate_t = ^H5L_iterate_t;

(* Callback for external link traversal *)
type
  H5L_elink_traverse_t = function(parent_file_name: PAnsiChar; parent_group_name: PAnsiChar; child_file_name: PAnsiChar; child_object_name: PAnsiChar; acc_flags: PCardinal; fapl_id: hid_t; op_data: Pointer): herr_t; cdecl;
  PH5L_elink_traverse_t = ^H5L_elink_traverse_t;

(* Flags for object copy (H5Ocopy) *)
const
  H5O_COPY_SHALLOW_HIERARCHY_FLAG = 1;  (* Copy only immediate members *)
  H5O_COPY_EXPAND_SOFT_LINK_FLAG = 2;  (* Expand soft links into new objects *)
  H5O_COPY_EXPAND_EXT_LINK_FLAG = 4;  (* Expand external links into new objects *)
  H5O_COPY_EXPAND_REFERENCE_FLAG = 8;  (* Copy objects that are pointed by references *)
  H5O_COPY_WITHOUT_ATTR_FLAG = 16;  (* Copy object without copying attributes *)
  H5O_COPY_PRESERVE_NULL_FLAG = 32;  (* Copy NULL messages (empty space) *)
  H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG = 64;  (* Merge committed datatypes in dest file *)
  H5O_COPY_ALL = 127;  (* All object copying flags (for internal checking) *)

(* Flags for shared message indexes.
 * Pass these flags in using the mesg_type_flags parameter in
 * H5P_set_shared_mesg_index.
 * (Developers: These flags correspond to object header message type IDs,
 * but we need to assign each kind of message to a different bit so that
 * one index can hold multiple types.)
 *)
const
  H5O_SHMESG_NONE_FLAG = 0;  (* No shared messages *)
  H5O_SHMESG_SDSPACE_FLAG = 1 shl 1;  (* Simple Dataspace Message. *)
  H5O_SHMESG_DTYPE_FLAG = 1 shl 3;  (* Datatype Message. *)
  H5O_SHMESG_FILL_FLAG = 1 shl 5;  (* Fill Value Message. *)
  H5O_SHMESG_PLINE_FLAG = 1 shl 11;  (* Filter pipeline message. *)
  H5O_SHMESG_ATTR_FLAG = 1 shl 12;  (* Attribute Message. *)
  H5O_SHMESG_ALL_FLAG = H5O_SHMESG_SDSPACE_FLAG or H5O_SHMESG_DTYPE_FLAG or H5O_SHMESG_FILL_FLAG or H5O_SHMESG_PLINE_FLAG or H5O_SHMESG_ATTR_FLAG;

(* Object header status flag definitions *)
const
  H5O_HDR_CHUNK0_SIZE = 3;  (* 2-bit field indicating # of bytes to store the size of chunk 0's data *)
  H5O_HDR_ATTR_CRT_ORDER_TRACKED = 4;  (* Attribute creation order is tracked *)
  H5O_HDR_ATTR_CRT_ORDER_INDEXED = 8;  (* Attribute creation order has index *)
  H5O_HDR_ATTR_STORE_PHASE_CHANGE = 16;  (* Non-default attribute storage phase change values stored *)
  H5O_HDR_STORE_TIMES = 32;  (* Store access, modification, change & birth times for object *)
  H5O_HDR_ALL_FLAGS = H5O_HDR_CHUNK0_SIZE or H5O_HDR_ATTR_CRT_ORDER_TRACKED or H5O_HDR_ATTR_CRT_ORDER_INDEXED or H5O_HDR_ATTR_STORE_PHASE_CHANGE or H5O_HDR_STORE_TIMES;

(* Maximum shared message values.  Number of indexes is 8 to allow room to add
 * new types of messages.
 *)
const
  H5O_SHMESG_MAX_NINDEXES = 8;
  H5O_SHMESG_MAX_LIST_SIZE = 5000;

(* Types of objects in file *)
type
  PH5O_type_t = ^H5O_type_t;
  H5O_type_t =
    (H5O_TYPE_UNKNOWN = -1,  (* Unknown object type *)
     H5O_TYPE_GROUP,  (* Object is a group *)
     H5O_TYPE_DATASET,  (* Object is a dataset *)
     H5O_TYPE_NAMED_DATATYPE,  (* Object is a named data type *)
     H5O_TYPE_NTYPES);  (* Number of different object types (must be last!) *)

(* Information struct for object header metadata (for H5Oget_info/H5Oget_info_by_name/H5Oget_info_by_idx) *)
type
  PH5O_hdr_info_t = ^H5O_hdr_info_t;
  PPH5O_hdr_info_t = ^PH5O_hdr_info_t;
  H5O_hdr_info_t = record
    version: Cardinal;  (* Version number of header format in file *)
    nmesgs: Cardinal;  (* Number of object header messages *)
    nchunks: Cardinal;  (* Number of object header chunks *)
    flags: Cardinal;  (* Object header status flags *)
    space: record
      total: hsize_t;  (* Total space for storing object header in file *)
      meta: hsize_t;  (* Space within header for object header metadata information *)
      mesg: hsize_t;  (* Space within header for actual message information *)
      free: hsize_t;  (* Free space within object header *)
    end;
    mesg: record
      present: uint64_t;  (* Flags to indicate presence of message type in header *)
      shared: uint64_t;  (* Flags to indicate message type is shared in header *)
    end;
  end;

(* Information struct for object (for H5Oget_info/H5Oget_info_by_name/H5Oget_info_by_idx) *)
type
  PH5O_info_t = ^H5O_info_t;
  PPH5O_info_t = ^PH5O_info_t;
  H5O_info_t = record
    fileno: Cardinal;  (* File number that object is located in *)
    addr: haddr_t;  (* Object address in file *)
    typ: H5O_type_t;  (* Basic object type (group, dataset, etc.) *)
    rc: Cardinal;  (* Reference count of object *)
    atime: time_t;  (* Access time *)
    mtime: time_t;  (* Modification time *)
    ctime: time_t;  (* Change time *)
    btime: time_t;  (* Birth time *)
    num_attrs: hsize_t;  (* # of attributes attached to object *)
    hdr: H5O_hdr_info_t;  (* Object header information *)
    meta_size: record
      obj: H5_ih_info_t;  (* v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets *)
      attr: H5_ih_info_t;  (* v2 B-tree & heap for attributes *)
    end;
  end;

(* Typedef for message creation indexes *)
type
  H5O_msg_crt_idx_t = uint32_t;
  PH5O_msg_crt_idx_t = ^H5O_msg_crt_idx_t;

(* Prototype for H5Ovisit/H5Ovisit_by_name() operator *)
type
  H5O_iterate_t = function(obj: hid_t; name: PAnsiChar; info: PH5O_info_t; op_data: Pointer): herr_t; cdecl;
  PH5O_iterate_t = ^H5O_iterate_t;
type
  PH5O_mcdt_search_ret_t = ^H5O_mcdt_search_ret_t;
  H5O_mcdt_search_ret_t =
    (H5O_MCDT_SEARCH_ERROR = -1,  (* Abort H5Ocopy *)
     H5O_MCDT_SEARCH_CONT,  (* Continue the global search of all committed datatypes in the destination file *)
     H5O_MCDT_SEARCH_STOP);  (* Stop the search, but continue copying.  The committed datatype will be copied but not merged. *)

(* Callback to invoke when completing the search for a matching committed datatype from the committed dtype list *)
type
  H5O_mcdt_search_cb_t = function(op_data: Pointer): H5O_mcdt_search_ret_t; cdecl;
  PH5O_mcdt_search_cb_t = ^H5O_mcdt_search_cb_t;

(*
 * These are the bits that can be passed to the `flags' argument of
 * H5Fcreate() and H5Fopen(). Use the bit-wise OR operator (|) to combine
 * them as needed.  As a side effect, they call H5check_version() to make sure
 * that the application is compiled with a version of the hdf5 header files
 * which are compatible with the library to which the application is linked.
 * We're assuming that these constants are used rather early in the hdf5
 * session.
 *
 * Note that H5F_ACC_DEBUG is deprecated (nonfuncational) but retained as a
 * symbol for backward compatibility.
 *)
const
  H5F_ACC_RDONLY = 0;  (* absence of rdwr => rd-only *)
  H5F_ACC_RDWR = 1;  (* open for read and write *)
  H5F_ACC_TRUNC = 2;  (* overwrite existing files *)
  H5F_ACC_EXCL = 4;  (* fail if file already exists *)
(* NOTE: 0x0008u was H5F_ACC_DEBUG, now deprecated *)
  H5F_ACC_CREAT = 16;  (* create non-existing files *)
  H5F_ACC_SWMR_WRITE = 32;
(*indicate that this file is
 * open for writing in a
 * single-writer/multi-reader (SWMR)
 * scenario.  Note that the
 * process(es) opening the file
 * for reading must open the file
 * with RDONLY access, and use
 * the special "SWMR_READ" access
 * flag. *)
  H5F_ACC_SWMR_READ = 64;
(*indicate that this file is
 * open for reading in a
 * single-writer/multi-reader (SWMR)
 * scenario.  Note that the
 * process(es) opening the file
 * for SWMR reading must also
 * open the file with the RDONLY
 * flag.  *)

(* Value passed to H5Pset_elink_acc_flags to cause flags to be taken from the
 * parent file. *)
const
  H5F_ACC_DEFAULT = 65535;  (* ignore setting on lapl *)

(* Flags for H5Fget_obj_count() & H5Fget_obj_ids() calls *)
const
  H5F_OBJ_FILE = 1;  (* File objects *)
  H5F_OBJ_DATASET = 2;  (* Dataset objects *)
  H5F_OBJ_GROUP = 4;  (* Group objects *)
  H5F_OBJ_DATATYPE = 8;  (* Named datatype objects *)
  H5F_OBJ_ATTR = 16;  (* Attribute objects *)
  H5F_OBJ_ALL = H5F_OBJ_FILE or H5F_OBJ_DATASET or H5F_OBJ_GROUP or H5F_OBJ_DATATYPE or H5F_OBJ_ATTR;
  H5F_OBJ_LOCAL = 32;  (* Restrict search to objects opened through current file ID *)
(* (as opposed to objects opened through any file ID accessing this file) *)
const
  H5F_FAMILY_DEFAULT = hsize_t(0);

(* The difference between a single file and a set of mounted files *)
type
  PH5F_scope_t = ^H5F_scope_t;
  H5F_scope_t =
    (H5F_SCOPE_LOCAL,  (* specified file handle only *)
     H5F_SCOPE_GLOBAL = 1);  (* entire virtual file *)

(* Unlimited file size for H5Pset_external() *)
const
  H5F_UNLIMITED = hsize_t(-1);

(* How does file close behave?
 * H5F_CLOSE_DEFAULT - Use the degree pre-defined by underlining VFL
 * H5F_CLOSE_WEAK    - file closes only after all opened objects are closed
 * H5F_CLOSE_SEMI    - if no opened objects, file is close; otherwise, file
                       close fails
 * H5F_CLOSE_STRONG  - if there are opened objects, close them first, then
                       close file
 *)
type
  PH5F_close_degree_t = ^H5F_close_degree_t;
  H5F_close_degree_t =
    (H5F_CLOSE_DEFAULT,
     H5F_CLOSE_WEAK = 1,
     H5F_CLOSE_SEMI = 2,
     H5F_CLOSE_STRONG = 3);

(* Current "global" information about file *)
type
  PH5F_info2_t = ^H5F_info2_t;
  PPH5F_info2_t = ^PH5F_info2_t;
  H5F_info2_t = record
    super: record
      version: Cardinal;  (* Superblock version # *)
      super_size: hsize_t;  (* Superblock size *)
      super_ext_size: hsize_t;  (* Superblock extension size *)
    end;
    free: record
      version: Cardinal;  (* Version # of file free space management *)
      meta_size: hsize_t;  (* Free space manager metadata size *)
      tot_space: hsize_t;  (* Amount of free space in the file *)
    end;
    sohm: record
      version: Cardinal;  (* Version # of shared object header info *)
      hdr_size: hsize_t;  (* Shared object header message header size *)
      msgs_info: H5_ih_info_t;  (* Shared object header message index & heap size *)
    end;
  end;

(*
 * Types of allocation requests. The values larger than H5FD_MEM_DEFAULT
 * should not change other than adding new types to the end. These numbers
 * might appear in files.
 *
 * Note: please change the log VFD flavors array if you change this
 * enumeration.
 *)
type
  PH5F_mem_t = ^H5F_mem_t;
  H5F_mem_t =
    (H5FD_MEM_NOLIST = -1,
(* Data should not appear in the free list.
 * Must be negative.
 *)
     H5FD_MEM_DEFAULT,
(* Value not yet set.  Can also be the
 * datatype set in a larger allocation
 * that will be suballocated by the library.
 * Must be zero.
 *)
     H5FD_MEM_SUPER = 1,  (* Superblock data *)
     H5FD_MEM_BTREE = 2,  (* B-tree data *)
     H5FD_MEM_DRAW = 3,  (* Raw data (content of datasets, etc.) *)
     H5FD_MEM_GHEAP = 4,  (* Global heap data *)
     H5FD_MEM_LHEAP = 5,  (* Local heap data *)
     H5FD_MEM_OHDR = 6,  (* Object header data *)

     H5FD_MEM_NTYPES);  (* Sentinel value - must be last *)

(* Free space section information *)
type
  PH5F_sect_info_t = ^H5F_sect_info_t;
  PPH5F_sect_info_t = ^PH5F_sect_info_t;
  H5F_sect_info_t = record
    addr: haddr_t;  (* Address of free space section *)
    size: hsize_t;  (* Size of free space section *)
  end;

(* Library's file format versions *)
type
  PH5F_libver_t = ^H5F_libver_t;
  H5F_libver_t =
    (H5F_LIBVER_EARLIEST,  (* Use the earliest possible format for storing objects *)
     H5F_LIBVER_LATEST);  (* Use the latest possible format available for storing objects *)

(* File space handling strategy *)
type
  PH5F_file_space_type_t = ^H5F_file_space_type_t;
  H5F_file_space_type_t =
    (H5F_FILE_SPACE_DEFAULT,  (* Default (or current) free space strategy setting *)
     H5F_FILE_SPACE_ALL_PERSIST = 1,  (* Persistent free space managers, aggregators, virtual file driver *)
     H5F_FILE_SPACE_ALL = 2,  (* Non-persistent free space managers, aggregators, virtual file driver *)
(* This is the library default *)
     H5F_FILE_SPACE_AGGR_VFD = 3,  (* Aggregators, Virtual file driver *)
     H5F_FILE_SPACE_VFD = 4,  (* Virtual file driver *)
     H5F_FILE_SPACE_NTYPES);  (* must be last *)

(* Data structure to report the collection of read retries for metadata items with checksum *)
(* Used by public routine H5Fget_metadata_read_retry_info() *)
const
  H5F_NUM_METADATA_READ_RETRY_TYPES = 21;
type
  PH5F_retry_info_t = ^H5F_retry_info_t;
  PPH5F_retry_info_t = ^PH5F_retry_info_t;
  H5F_retry_info_t = record
    nbins: Cardinal;
    retries: array[0..H5F_NUM_METADATA_READ_RETRY_TYPES - 1] of Puint32_t;
  end;

(* Callback for H5Pset_object_flush_cb() in a file access property list *)
type
  H5F_flush_cb_t = function(object_id: hid_t; udata: Pointer): herr_t; cdecl;
  PH5F_flush_cb_t = ^H5F_flush_cb_t;

(* Information struct for attribute (for H5Aget_info/H5Aget_info_by_idx) *)
type
  PH5A_info_t = ^H5A_info_t;
  PPH5A_info_t = ^PH5A_info_t;
  H5A_info_t = record
    corder_valid: hbool_t;  (* Indicate if creation order is valid *)
    corder: H5O_msg_crt_idx_t;  (* Creation order *)
    cset: H5T_cset_t;  (* Character set of attribute name *)
    data_size: hsize_t;  (* Size of raw data *)
  end;

(* Typedef for H5Aiterate2() callbacks *)
type
  H5A_operator2_t = function(location_id: hid_t; attr_name: PAnsiChar; ainfo: PH5A_info_t; op_data: Pointer): herr_t; cdecl;
  PH5A_operator2_t = ^H5A_operator2_t;
const
  H5_HAVE_VFL = 1;  (* define a convenient app feature test *)
  H5FD_VFD_DEFAULT = 0;  (* Default VFL driver value *)

(* Types of allocation requests: see H5Fpublic.h  *)
type
  H5FD_mem_t = H5F_mem_t;
  PH5FD_mem_t = ^H5FD_mem_t;

(* Map "fractal heap" header blocks to 'ohdr' type file memory, since its
 * a fair amount of work to add a new kind of file memory and they are similar
 * enough to object headers and probably too minor to deserve their own type.
 *
 * Map "fractal heap" indirect blocks to 'ohdr' type file memory, since they
 * are similar to fractal heap header blocks.
 *
 * Map "fractal heap" direct blocks to 'lheap' type file memory, since they
 * will be replacing local heaps.
 *
 * Map "fractal heap" 'huge' objects to 'draw' type file memory, since they
 * represent large objects that are directly stored in the file.
 *
 *      -QAK
 *)
const
  H5FD_MEM_FHEAP_HDR = H5FD_MEM_OHDR;
  H5FD_MEM_FHEAP_IBLOCK = H5FD_MEM_OHDR;
  H5FD_MEM_FHEAP_DBLOCK = H5FD_MEM_LHEAP;
  H5FD_MEM_FHEAP_HUGE_OBJ = H5FD_MEM_DRAW;

(* Map "free space" header blocks to 'ohdr' type file memory, since its
 * a fair amount of work to add a new kind of file memory and they are similar
 * enough to object headers and probably too minor to deserve their own type.
 *
 * Map "free space" serialized sections to 'lheap' type file memory, since they
 * are similar enough to local heap info.
 *
 *      -QAK
 *)
const
  H5FD_MEM_FSPACE_HDR = H5FD_MEM_OHDR;
  H5FD_MEM_FSPACE_SINFO = H5FD_MEM_LHEAP;

(* Map "shared object header message" master table to 'ohdr' type file memory,
 * since its a fair amount of work to add a new kind of file memory and they are
 * similar enough to object headers and probably too minor to deserve their own
 * type.
 *
 * Map "shared object header message" indices to 'btree' type file memory,
 * since they are similar enough to B-tree nodes.
 *
 *      -QAK
 *)
const
  H5FD_MEM_SOHM_TABLE = H5FD_MEM_OHDR;
  H5FD_MEM_SOHM_INDEX = H5FD_MEM_BTREE;

(* Map "extensible array" header blocks to 'ohdr' type file memory, since its
 * a fair amount of work to add a new kind of file memory and they are similar
 * enough to object headers and probably too minor to deserve their own type.
 *
 * Map "extensible array" index blocks to 'ohdr' type file memory, since they
 * are similar to extensible array header blocks.
 *
 * Map "extensible array" super blocks to 'btree' type file memory, since they
 * are similar enough to B-tree nodes.
 *
 * Map "extensible array" data blocks & pages to 'lheap' type file memory, since
 * they are similar enough to local heap info.
 *
 *      -QAK
 *)
const
  H5FD_MEM_EARRAY_HDR = H5FD_MEM_OHDR;
  H5FD_MEM_EARRAY_IBLOCK = H5FD_MEM_OHDR;
  H5FD_MEM_EARRAY_SBLOCK = H5FD_MEM_BTREE;
  H5FD_MEM_EARRAY_DBLOCK = H5FD_MEM_LHEAP;
  H5FD_MEM_EARRAY_DBLK_PAGE = H5FD_MEM_LHEAP;

(* Map "fixed array" header blocks to 'ohdr' type file memory, since its
 * a fair amount of work to add a new kind of file memory and they are similar
 * enough to object headers and probably too minor to deserve their own type.
 *
 * Map "fixed array" data blocks & pages to 'lheap' type file memory, since
 * they are similar enough to local heap info.
 *
 *)
const
  H5FD_MEM_FARRAY_HDR = H5FD_MEM_OHDR;
  H5FD_MEM_FARRAY_DBLOCK = H5FD_MEM_LHEAP;
  H5FD_MEM_FARRAY_DBLK_PAGE = H5FD_MEM_LHEAP;

(* Define VFL driver features that can be enabled on a per-driver basis *)
(* These are returned with the 'query' function pointer in H5FD_class_t *)
(*
     * Defining H5FD_FEAT_AGGREGATE_METADATA for a VFL driver means that
     * the library will attempt to allocate a larger block for metadata and
     * then sub-allocate each metadata request from that larger block.
     *)
const
  H5FD_FEAT_AGGREGATE_METADATA = 1;
(*
 * Defining H5FD_FEAT_ACCUMULATE_METADATA for a VFL driver means that
 * the library will attempt to cache metadata as it is written to the file
 * and build up a larger block of metadata to eventually pass to the VFL
 * 'write' routine.
 *
 * Distinguish between updating the metadata accumulator on writes and
 * reads.  This is particularly (perhaps only, even) important for MPI-I/O
 * where we guarantee that writes are collective, but reads may not be.
 * If we were to allow the metadata accumulator to be written during a
 * read operation, the application would hang.
 *)
  H5FD_FEAT_ACCUMULATE_METADATA_WRITE = 2;
  H5FD_FEAT_ACCUMULATE_METADATA_READ = 4;
  H5FD_FEAT_ACCUMULATE_METADATA = H5FD_FEAT_ACCUMULATE_METADATA_WRITE or H5FD_FEAT_ACCUMULATE_METADATA_READ;
(*
 * Defining H5FD_FEAT_DATA_SIEVE for a VFL driver means that
 * the library will attempt to cache raw data as it is read from/written to
 * a file in a "data seive" buffer.  See Rajeev Thakur's papers:
 *  http://www.mcs.anl.gov/~thakur/papers/romio-coll.ps.gz
 *  http://www.mcs.anl.gov/~thakur/papers/mpio-high-perf.ps.gz
 *)
  H5FD_FEAT_DATA_SIEVE = 8;
(*
 * Defining H5FD_FEAT_AGGREGATE_SMALLDATA for a VFL driver means that
 * the library will attempt to allocate a larger block for "small" raw data
 * and then sub-allocate "small" raw data requests from that larger block.
 *)
  H5FD_FEAT_AGGREGATE_SMALLDATA = 16;
(*
 * Defining H5FD_FEAT_IGNORE_DRVRINFO for a VFL driver means that
 * the library will ignore the driver info that is encoded in the file
 * for the VFL driver.  (This will cause the driver info to be eliminated
 * from the file when it is flushed/closed, if the file is opened R/W).
 *)
  H5FD_FEAT_IGNORE_DRVRINFO = 32;
(*
 * Defining the H5FD_FEAT_DIRTY_DRVRINFO_LOAD for a VFL driver means that
 * the library will mark the driver info dirty when the file is opened
 * R/W.  This will cause the driver info to be re-encoded when the file
 * is flushed/closed.
 *)
  H5FD_FEAT_DIRTY_DRVRINFO_LOAD = 64;
(*
 * Defining H5FD_FEAT_POSIX_COMPAT_HANDLE for a VFL driver means that
 * the handle for the VFD (returned with the 'get_handle' callback) is
 * of type 'int' and is compatible with POSIX I/O calls.
 *)
  H5FD_FEAT_POSIX_COMPAT_HANDLE = 128;
(*
 * Defining H5FD_FEAT_HAS_MPI for a VFL driver means that
 * the driver makes use of MPI communication and code may retrieve
 * communicator/rank information from it
 *)
  H5FD_FEAT_HAS_MPI = 256;
(*
 * Defining the H5FD_FEAT_ALLOCATE_EARLY for a VFL driver will force
 * the library to use the H5D_ALLOC_TIME_EARLY on dataset create
 * instead of the default H5D_ALLOC_TIME_LATE
 *)
  H5FD_FEAT_ALLOCATE_EARLY = 512;
(*
 * Defining H5FD_FEAT_ALLOW_FILE_IMAGE for a VFL driver means that
 * the driver is able to use a file image in the fapl as the initial
 * contents of a file.
 *)
  H5FD_FEAT_ALLOW_FILE_IMAGE = 1024;
(*
 * Defining H5FD_FEAT_CAN_USE_FILE_IMAGE_CALLBACKS for a VFL driver
 * means that the driver is able to use callbacks to make a copy of the
 * image to store in memory.
 *)
  H5FD_FEAT_CAN_USE_FILE_IMAGE_CALLBACKS = 2048;
(*
 * Defining H5FD_FEAT_SUPPORTS_SWMR_IO for a VFL driver means that the
 * driver supports the single-writer/multiple-readers I/O pattern.
 *)
  H5FD_FEAT_SUPPORTS_SWMR_IO = 4096;

(* Class information for each file driver *)
type
  PH5FD_class_t = ^H5FD_class_t;
  PPH5FD_class_t = ^PH5FD_class_t;
  H5FD_class_t = record
    name: PAnsiChar;
    maxaddr: haddr_t;
    fc_degree: H5F_close_degree_t;
    terminate: function: herr_t; cdecl;
    sb_size: function(file_: Pointer {PH5FD_t}): hsize_t; cdecl;
    sb_encode: function(file_: Pointer {PH5FD_t}; name: PAnsiChar; p: PByte): herr_t; cdecl;
    sb_decode: function(f: Pointer {PH5FD_t}; name: PAnsiChar; p: PByte): herr_t; cdecl;
    fapl_size: size_t;
    fapl_get: function(file_: Pointer {PH5FD_t}): Pointer; cdecl;
    fapl_copy: function(fapl: Pointer): Pointer; cdecl;
    fapl_free: function(fapl: Pointer): herr_t; cdecl;
    dxpl_size: size_t;
    dxpl_copy: function(dxpl: Pointer): Pointer; cdecl;
    dxpl_free: function(dxpl: Pointer): herr_t; cdecl;
    open: function(name: PAnsiChar; flags: Cardinal; fapl: hid_t; maxaddr: haddr_t): Pointer {PH5FD_t}; cdecl;
    close: function(file_: Pointer {PH5FD_t}): herr_t; cdecl;
    cmp: function(f1: Pointer {PH5FD_t}; f2: Pointer {PH5FD_t}): Integer; cdecl;
    query: function(f1: Pointer {PH5FD_t}; flags: PCardinal): herr_t; cdecl;
    get_type_map: function(file_: Pointer {PH5FD_t}; type_map: PH5FD_mem_t): herr_t; cdecl;
    alloc: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t; dxpl_id: hid_t; size: hsize_t): haddr_t; cdecl;
    free: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t; dxpl_id: hid_t; addr: haddr_t; size: hsize_t): herr_t; cdecl;
    get_eoa: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t): haddr_t; cdecl;
    set_eoa: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t; addr: haddr_t): herr_t; cdecl;
    get_eof: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t): haddr_t; cdecl;
    get_handle: function(file_: Pointer {PH5FD_t}; fapl: hid_t; file_handle: PPointer): herr_t; cdecl;
    read: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t; dxpl: hid_t; addr: haddr_t; size: size_t; buffer: Pointer): herr_t; cdecl;
    write: function(file_: Pointer {PH5FD_t}; typ: H5FD_mem_t; dxpl: hid_t; addr: haddr_t; size: size_t; buffer: Pointer): herr_t; cdecl;
    flush: function(file_: Pointer {PH5FD_t}; dxpl_id: hid_t; closing: Cardinal): herr_t; cdecl;
    truncate: function(file_: Pointer {PH5FD_t}; dxpl_id: hid_t; closing: hbool_t): herr_t; cdecl;
    lock: function(file_: Pointer {PH5FD_t}; rw: hbool_t): herr_t; cdecl;
    unlock: function(file_: Pointer {PH5FD_t}): herr_t; cdecl;
    fl_map: array[H5FD_MEM_DEFAULT..Pred(H5FD_MEM_NTYPES)] of H5FD_mem_t;
  end;

(* A free list is a singly-linked list of address/size pairs. *)
type
  PH5FD_free_t = ^H5FD_free_t;
  PPH5FD_free_t = ^PH5FD_free_t;
  H5FD_free_t = record
    addr: haddr_t;
    size: hsize_t;
  end;

(*
 * The main datatype for each driver. Public fields common to all drivers
 * are declared here and the driver appends private fields in memory.
 *)
type
  PH5FD_t = ^H5FD_t;
  PPH5FD_t = ^PH5FD_t;
  H5FD_t = record
    driver_id: hid_t;  (* driver ID for this file *)
    cls: PH5FD_class_t;  (* constant class info *)
    fileno: Cardinal;  (* File 'serial' number *)
    feature_flags: Cardinal;  (* VFL Driver feature Flags *)
    maxaddr: haddr_t;  (* For this file, overrides class *)
    base_addr: haddr_t;  (* Base address for HDF5 data w/in file *)
    swmr_read: hbool_t;  (* Whether the file is open for SWMR read access *)
    threshold: hsize_t;  (* Threshold for alignment *)
    alignment: hsize_t;  (* Allocation alignment *)
  end;

(* Define enum for the source of file image callbacks *)
type
  PH5FD_file_image_op_t = ^H5FD_file_image_op_t;
  H5FD_file_image_op_t =
    (H5FD_FILE_IMAGE_OP_NO_OP,
     H5FD_FILE_IMAGE_OP_PROPERTY_LIST_SET,
     H5FD_FILE_IMAGE_OP_PROPERTY_LIST_COPY,
     H5FD_FILE_IMAGE_OP_PROPERTY_LIST_GET,
     H5FD_FILE_IMAGE_OP_PROPERTY_LIST_CLOSE,
     H5FD_FILE_IMAGE_OP_FILE_OPEN,
     H5FD_FILE_IMAGE_OP_FILE_RESIZE,
     H5FD_FILE_IMAGE_OP_FILE_CLOSE);

(* Define structure to hold file image callbacks *)
type
  PH5FD_file_image_callbacks_t = ^H5FD_file_image_callbacks_t;
  PPH5FD_file_image_callbacks_t = ^PH5FD_file_image_callbacks_t;
  H5FD_file_image_callbacks_t = record
    image_malloc: function(size: size_t; file_image_op: H5FD_file_image_op_t; udata: Pointer): Pointer; cdecl;
    image_memcpy: function(dest: Pointer; src: Pointer; size: size_t; file_image_op: H5FD_file_image_op_t; udata: Pointer): Pointer; cdecl;
    image_realloc: function(ptr: Pointer; size: size_t; file_image_op: H5FD_file_image_op_t; udata: Pointer): Pointer; cdecl;
    image_free: function(ptr: Pointer; file_image_op: H5FD_file_image_op_t; udata: Pointer): herr_t; cdecl;
    udata_copy: function(udata: Pointer): Pointer; cdecl;
    udata_free: function(udata: Pointer): herr_t; cdecl;
    udata: Pointer;
  end;

(* Types of link storage for groups *)
type
  PH5G_storage_type_t = ^H5G_storage_type_t;
  H5G_storage_type_t =
    (H5G_STORAGE_TYPE_UNKNOWN = -1,  (* Unknown link storage type *)
     H5G_STORAGE_TYPE_SYMBOL_TABLE,  (* Links in group are stored with a "symbol table" *)
(* (this is sometimes called "old-style" groups) *)
     H5G_STORAGE_TYPE_COMPACT,  (* Links are stored in object header *)
     H5G_STORAGE_TYPE_DENSE);  (* Links are stored in fractal heap & indexed with v2 B-tree *)

(* Information struct for group (for H5Gget_info/H5Gget_info_by_name/H5Gget_info_by_idx) *)
type
  PH5G_info_t = ^H5G_info_t;
  PPH5G_info_t = ^PH5G_info_t;
  H5G_info_t = record
    storage_type: H5G_storage_type_t;  (* Type of storage for links in group *)
    nlinks: hsize_t;  (* Number of links in group *)
    max_corder: int64_t;  (* Current max. creation order value for group *)
    mounted: hbool_t;  (* Whether group has a file mounted on it *)
  end;

(*
 * Reference types allowed.
 *)
type
  PH5R_type_t = ^H5R_type_t;
  H5R_type_t =
    (H5R_BADTYPE = -1,  (* invalid Reference Type *)
     H5R_OBJECT,  (* Object reference *)
     H5R_DATASET_REGION,  (* Dataset Region Reference *)
     H5R_MAXTYPE);  (* highest type (Invalid as true type) *)

(* Note! Be careful with the sizes of the references because they should really
 * depend on the run-time values in the file.  Unfortunately, the arrays need
 * to be defined at compile-time, so we have to go with the worst case sizes for
 * them.  -QAK
 *)
const
  H5R_OBJ_REF_BUF_SIZE = SizeOf(haddr_t);
(* Object reference structure for user's code *)
type
  hobj_ref_t = haddr_t;
  Phobj_ref_t = ^hobj_ref_t;
const
  H5R_DSET_REG_REF_BUF_SIZE = SizeOf(haddr_t)+4;
(* 4 is used instead of sizeof(int) to permit portability between
   the Crays and other machines (the heap ID is always encoded as an int32 anyway)
*)
(* Dataset Region reference structure for user's code *)
type
  hdset_reg_ref_t = array[0..H5R_DSET_REG_REF_BUF_SIZE - 1] of Byte;
  Phdset_reg_ref_t = ^hdset_reg_ref_t;
(* Needs to be large enough to store largest haddr_t in a worst case machine (ie. 8 bytes currently) plus an int *)

(* Common creation order flags (for links in groups and attributes on objects) *)
const
  H5P_CRT_ORDER_TRACKED = 1;
  H5P_CRT_ORDER_INDEXED = 2;

(* Default value for all property list classes *)
const
  H5P_DEFAULT = hid_t(0);

(* Define property list class callback function pointer types *)
type
  H5P_cls_create_func_t = function(prop_id: hid_t; create_data: Pointer): herr_t; cdecl;
  PH5P_cls_create_func_t = ^H5P_cls_create_func_t;
  H5P_cls_copy_func_t = function(new_prop_id: hid_t; old_prop_id: hid_t; copy_data: Pointer): herr_t; cdecl;
  PH5P_cls_copy_func_t = ^H5P_cls_copy_func_t;
  H5P_cls_close_func_t = function(prop_id: hid_t; close_data: Pointer): herr_t; cdecl;
  PH5P_cls_close_func_t = ^H5P_cls_close_func_t;

(* Define property list callback function pointer types *)
type
  H5P_prp_cb1_t = function(name: PAnsiChar; size: size_t; value: Pointer): herr_t; cdecl;
  PH5P_prp_cb1_t = ^H5P_prp_cb1_t;
  H5P_prp_cb2_t = function(prop_id: hid_t; name: PAnsiChar; size: size_t; value: Pointer): herr_t; cdecl;
  PH5P_prp_cb2_t = ^H5P_prp_cb2_t;
  H5P_prp_create_func_t = H5P_prp_cb1_t;
  PH5P_prp_create_func_t = ^H5P_prp_create_func_t;
  H5P_prp_set_func_t = H5P_prp_cb2_t;
  PH5P_prp_set_func_t = ^H5P_prp_set_func_t;
  H5P_prp_get_func_t = H5P_prp_cb2_t;
  PH5P_prp_get_func_t = ^H5P_prp_get_func_t;
  H5P_prp_encode_func_t = function(value: Pointer; buf: PPointer; size: Psize_t): herr_t; cdecl;
  PH5P_prp_encode_func_t = ^H5P_prp_encode_func_t;
  H5P_prp_decode_func_t = function(buf: PPointer; value: Pointer): herr_t; cdecl;
  PH5P_prp_decode_func_t = ^H5P_prp_decode_func_t;
  H5P_prp_delete_func_t = H5P_prp_cb2_t;
  PH5P_prp_delete_func_t = ^H5P_prp_delete_func_t;
  H5P_prp_copy_func_t = H5P_prp_cb1_t;
  PH5P_prp_copy_func_t = ^H5P_prp_copy_func_t;
  H5P_prp_compare_func_t = function(value1: Pointer; value2: Pointer; size: size_t): Integer; cdecl;
  PH5P_prp_compare_func_t = ^H5P_prp_compare_func_t;
  H5P_prp_close_func_t = H5P_prp_cb1_t;
  PH5P_prp_close_func_t = ^H5P_prp_close_func_t;

(* Define property list iteration function type *)
type
  H5P_iterate_t = function(id: hid_t; name: PAnsiChar; iter_data: Pointer): herr_t; cdecl;
  PH5P_iterate_t = ^H5P_iterate_t;

(* Actual IO mode property *)
type
  PH5D_mpio_actual_chunk_opt_mode_t = ^H5D_mpio_actual_chunk_opt_mode_t;
  H5D_mpio_actual_chunk_opt_mode_t =
(* The default value, H5D_MPIO_NO_CHUNK_OPTIMIZATION, is used for all I/O
 * operations that do not use chunk optimizations, including non-collective
 * I/O and contiguous collective I/O.
 *)
    (H5D_MPIO_NO_CHUNK_OPTIMIZATION,
     H5D_MPIO_LINK_CHUNK,
     H5D_MPIO_MULTI_CHUNK);
type
  PH5D_mpio_actual_io_mode_t = ^H5D_mpio_actual_io_mode_t;
  H5D_mpio_actual_io_mode_t =
(* The following four values are conveniently defined as a bit field so that
 * we can switch from the default to indpendent or collective and then to
 * mixed without having to check the original value.
 *
 * NO_COLLECTIVE means that either collective I/O wasn't requested or that
 * no I/O took place.
 *
 * CHUNK_INDEPENDENT means that collective I/O was requested, but the
 * chunk optimization scheme chose independent I/O for each chunk.
 *)
    (H5D_MPIO_NO_COLLECTIVE,
     H5D_MPIO_CHUNK_INDEPENDENT = 1,
     H5D_MPIO_CHUNK_COLLECTIVE = 2,
     H5D_MPIO_CHUNK_MIXED,

(* The contiguous case is separate from the bit field. *)
     H5D_MPIO_CONTIGUOUS_COLLECTIVE = 4);

(* Broken collective IO property *)
type
  PH5D_mpio_no_collective_cause_t = ^H5D_mpio_no_collective_cause_t;
  H5D_mpio_no_collective_cause_t =
    (H5D_MPIO_COLLECTIVE,
     H5D_MPIO_SET_INDEPENDENT = 1,
     H5D_MPIO_DATATYPE_CONVERSION = 2,
     H5D_MPIO_DATA_TRANSFORMS = 4,
     H5D_MPIO_MPI_OPT_TYPES_ENV_VAR_DISABLED = 8,
     H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES = 16,
     H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET = 32,
     H5D_MPIO_FILTERS = 64);

type
  THDF5Dll = class
  private
  type
    TH5open = function: herr_t; cdecl;
    TH5close = function: herr_t; cdecl;
    TH5dont_atexit = function: herr_t; cdecl;
    TH5garbage_collect = function: herr_t; cdecl;
    TH5set_free_list_limits = function(reg_global_lim: Integer; reg_list_lim: Integer; arr_global_lim: Integer; arr_list_lim: Integer; blk_global_lim: Integer; blk_list_lim: Integer): herr_t; cdecl;
    TH5get_libversion = function(majnum: PCardinal; minnum: PCardinal; relnum: PCardinal): herr_t; cdecl;
    TH5check_version = function(majnum: Cardinal; minnum: Cardinal; relnum: Cardinal): herr_t; cdecl;
    TH5is_library_threadsafe = function(is_ts: Phbool_t): herr_t; cdecl;
    TH5free_memory = function(mem: Pointer): herr_t; cdecl;
    TH5allocate_memory = function(size: size_t; clear: hbool_t): Pointer; cdecl;
    TH5resize_memory = function(mem: Pointer; size: size_t): Pointer; cdecl;
    TH5Iregister = function(typ: H5I_type_t; obj: Pointer): hid_t; cdecl;
    TH5Iobject_verify = function(id: hid_t; id_type: H5I_type_t): Pointer; cdecl;
    TH5Iremove_verify = function(id: hid_t; id_type: H5I_type_t): Pointer; cdecl;
    TH5Iget_type = function(id: hid_t): H5I_type_t; cdecl;
    TH5Iget_file_id = function(id: hid_t): hid_t; cdecl;
    TH5Iget_name = function(id: hid_t; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Iinc_ref = function(id: hid_t): Integer; cdecl;
    TH5Idec_ref = function(id: hid_t): Integer; cdecl;
    TH5Iget_ref = function(id: hid_t): Integer; cdecl;
    TH5Iregister_type = function(hash_size: size_t; reserved: Cardinal; free_func: H5I_free_t): H5I_type_t; cdecl;
    TH5Iclear_type = function(typ: H5I_type_t; force: hbool_t): herr_t; cdecl;
    TH5Idestroy_type = function(typ: H5I_type_t): herr_t; cdecl;
    TH5Iinc_type_ref = function(typ: H5I_type_t): Integer; cdecl;
    TH5Idec_type_ref = function(typ: H5I_type_t): Integer; cdecl;
    TH5Iget_type_ref = function(typ: H5I_type_t): Integer; cdecl;
    TH5Isearch = function(typ: H5I_type_t; func: H5I_search_func_t; key: Pointer): Pointer; cdecl;
    TH5Inmembers = function(typ: H5I_type_t; num_members: Phsize_t): herr_t; cdecl;
    TH5Itype_exists = function(typ: H5I_type_t): htri_t; cdecl;
    TH5Iis_valid = function(id: hid_t): htri_t; cdecl;
    TH5Zregister = function(cls: Pointer): herr_t; cdecl;
    TH5Zunregister = function(id: H5Z_filter_t): herr_t; cdecl;
    TH5Zfilter_avail = function(id: H5Z_filter_t): htri_t; cdecl;
    TH5Zget_filter_info = function(filter: H5Z_filter_t; filter_config_flags: PCardinal): herr_t; cdecl;
    TH5PLset_loading_state = function(plugin_type: Cardinal): herr_t; cdecl;
    TH5PLget_loading_state = function(plugin_type: PCardinal): herr_t; cdecl;
    TH5Tcreate = function(typ: H5T_class_t; size: size_t): hid_t; cdecl;
    TH5Tcopy = function(type_id: hid_t): hid_t; cdecl;
    TH5Tclose = function(type_id: hid_t): herr_t; cdecl;
    TH5Tequal = function(type1_id: hid_t; type2_id: hid_t): htri_t; cdecl;
    TH5Tlock = function(type_id: hid_t): herr_t; cdecl;
    TH5Tcommit2 = function(loc_id: hid_t; name: PAnsiChar; type_id: hid_t; lcpl_id: hid_t; tcpl_id: hid_t; tapl_id: hid_t): herr_t; cdecl;
    TH5Topen2 = function(loc_id: hid_t; name: PAnsiChar; tapl_id: hid_t): hid_t; cdecl;
    TH5Tcommit_anon = function(loc_id: hid_t; type_id: hid_t; tcpl_id: hid_t; tapl_id: hid_t): herr_t; cdecl;
    TH5Tget_create_plist = function(type_id: hid_t): hid_t; cdecl;
    TH5Tcommitted = function(type_id: hid_t): htri_t; cdecl;
    TH5Tencode = function(obj_id: hid_t; buf: Pointer; nalloc: Psize_t): herr_t; cdecl;
    TH5Tdecode = function(buf: Pointer): hid_t; cdecl;
    TH5Tflush = function(type_id: hid_t): herr_t; cdecl;
    TH5Trefresh = function(type_id: hid_t): herr_t; cdecl;
    TH5Tinsert = function(parent_id: hid_t; name: PAnsiChar; offset: size_t; member_id: hid_t): herr_t; cdecl;
    TH5Tpack = function(type_id: hid_t): herr_t; cdecl;
    TH5Tenum_create = function(base_id: hid_t): hid_t; cdecl;
    TH5Tenum_insert = function(typ: hid_t; name: PAnsiChar; value: Pointer): herr_t; cdecl;
    TH5Tenum_nameof = function(typ: hid_t; value: Pointer; name: PAnsiChar; size: size_t): herr_t; cdecl;
    TH5Tenum_valueof = function(typ: hid_t; name: PAnsiChar; value: Pointer): herr_t; cdecl;
    TH5Tvlen_create = function(base_id: hid_t): hid_t; cdecl;
    TH5Tarray_create2 = function(base_id: hid_t; ndims: Cardinal; dim: Phsize_t): hid_t; cdecl;
    TH5Tget_array_ndims = function(type_id: hid_t): Integer; cdecl;
    TH5Tget_array_dims2 = function(type_id: hid_t; dims: Phsize_t): Integer; cdecl;
    TH5Tset_tag = function(typ: hid_t; tag: PAnsiChar): herr_t; cdecl;
    TH5Tget_tag = function(typ: hid_t): PAnsiChar; cdecl;
    TH5Tget_super = function(typ: hid_t): hid_t; cdecl;
    TH5Tget_class = function(type_id: hid_t): H5T_class_t; cdecl;
    TH5Tdetect_class = function(type_id: hid_t; cls: H5T_class_t): htri_t; cdecl;
    TH5Tget_size = function(type_id: hid_t): size_t; cdecl;
    TH5Tget_order = function(type_id: hid_t): H5T_order_t; cdecl;
    TH5Tget_precision = function(type_id: hid_t): size_t; cdecl;
    TH5Tget_offset = function(type_id: hid_t): Integer; cdecl;
    TH5Tget_pad = function(type_id: hid_t; lsb: PH5T_pad_t; msb: PH5T_pad_t): herr_t; cdecl;
    TH5Tget_sign = function(type_id: hid_t): H5T_sign_t; cdecl;
    TH5Tget_fields = function(type_id: hid_t; spos: Psize_t; epos: Psize_t; esize: Psize_t; mpos: Psize_t; msize: Psize_t): herr_t; cdecl;
    TH5Tget_ebias = function(type_id: hid_t): size_t; cdecl;
    TH5Tget_norm = function(type_id: hid_t): H5T_norm_t; cdecl;
    TH5Tget_inpad = function(type_id: hid_t): H5T_pad_t; cdecl;
    TH5Tget_strpad = function(type_id: hid_t): H5T_str_t; cdecl;
    TH5Tget_nmembers = function(type_id: hid_t): Integer; cdecl;
    TH5Tget_member_name = function(type_id: hid_t; membno: Cardinal): PAnsiChar; cdecl;
    TH5Tget_member_index = function(type_id: hid_t; name: PAnsiChar): Integer; cdecl;
    TH5Tget_member_offset = function(type_id: hid_t; membno: Cardinal): size_t; cdecl;
    TH5Tget_member_class = function(type_id: hid_t; membno: Cardinal): H5T_class_t; cdecl;
    TH5Tget_member_type = function(type_id: hid_t; membno: Cardinal): hid_t; cdecl;
    TH5Tget_member_value = function(type_id: hid_t; membno: Cardinal; value: Pointer): herr_t; cdecl;
    TH5Tget_cset = function(type_id: hid_t): H5T_cset_t; cdecl;
    TH5Tis_variable_str = function(type_id: hid_t): htri_t; cdecl;
    TH5Tget_native_type = function(type_id: hid_t; direction: H5T_direction_t): hid_t; cdecl;
    TH5Tset_size = function(type_id: hid_t; size: size_t): herr_t; cdecl;
    TH5Tset_order = function(type_id: hid_t; order: H5T_order_t): herr_t; cdecl;
    TH5Tset_precision = function(type_id: hid_t; prec: size_t): herr_t; cdecl;
    TH5Tset_offset = function(type_id: hid_t; offset: size_t): herr_t; cdecl;
    TH5Tset_pad = function(type_id: hid_t; lsb: H5T_pad_t; msb: H5T_pad_t): herr_t; cdecl;
    TH5Tset_sign = function(type_id: hid_t; sign: H5T_sign_t): herr_t; cdecl;
    TH5Tset_fields = function(type_id: hid_t; spos: size_t; epos: size_t; esize: size_t; mpos: size_t; msize: size_t): herr_t; cdecl;
    TH5Tset_ebias = function(type_id: hid_t; ebias: size_t): herr_t; cdecl;
    TH5Tset_norm = function(type_id: hid_t; norm: H5T_norm_t): herr_t; cdecl;
    TH5Tset_inpad = function(type_id: hid_t; pad: H5T_pad_t): herr_t; cdecl;
    TH5Tset_cset = function(type_id: hid_t; cset: H5T_cset_t): herr_t; cdecl;
    TH5Tset_strpad = function(type_id: hid_t; strpad: H5T_str_t): herr_t; cdecl;
    TH5Tregister = function(pers: H5T_pers_t; name: PAnsiChar; src_id: hid_t; dst_id: hid_t; func: H5T_conv_t): herr_t; cdecl;
    TH5Tunregister = function(pers: H5T_pers_t; name: PAnsiChar; src_id: hid_t; dst_id: hid_t; func: H5T_conv_t): herr_t; cdecl;
    TH5Tfind = function(src_id: hid_t; dst_id: hid_t; pcdata: PPH5T_cdata_t): H5T_conv_t; cdecl;
    TH5Tcompiler_conv = function(src_id: hid_t; dst_id: hid_t): htri_t; cdecl;
    TH5Tconvert = function(src_id: hid_t; dst_id: hid_t; nelmts: size_t; buf: Pointer; background: Pointer; plist_id: hid_t): herr_t; cdecl;
    TH5Dcreate2 = function(loc_id: hid_t; name: PAnsiChar; type_id: hid_t; space_id: hid_t; lcpl_id: hid_t; dcpl_id: hid_t; dapl_id: hid_t): hid_t; cdecl;
    TH5Dcreate_anon = function(file_id: hid_t; type_id: hid_t; space_id: hid_t; plist_id: hid_t; dapl_id: hid_t): hid_t; cdecl;
    TH5Dopen2 = function(file_id: hid_t; name: PAnsiChar; dapl_id: hid_t): hid_t; cdecl;
    TH5Dclose = function(dset_id: hid_t): herr_t; cdecl;
    TH5Dget_space = function(dset_id: hid_t): hid_t; cdecl;
    TH5Dget_space_status = function(dset_id: hid_t; allocation: PH5D_space_status_t): herr_t; cdecl;
    TH5Dget_type = function(dset_id: hid_t): hid_t; cdecl;
    TH5Dget_create_plist = function(dset_id: hid_t): hid_t; cdecl;
    TH5Dget_access_plist = function(dset_id: hid_t): hid_t; cdecl;
    TH5Dget_storage_size = function(dset_id: hid_t): hsize_t; cdecl;
    TH5Dget_offset = function(dset_id: hid_t): haddr_t; cdecl;
    TH5Dread = function(dset_id: hid_t; mem_type_id: hid_t; mem_space_id: hid_t; file_space_id: hid_t; plist_id: hid_t; buf: Pointer): herr_t; cdecl;
    TH5Dwrite = function(dset_id: hid_t; mem_type_id: hid_t; mem_space_id: hid_t; file_space_id: hid_t; plist_id: hid_t; buf: Pointer): herr_t; cdecl;
    TH5Diterate = function(buf: Pointer; type_id: hid_t; space_id: hid_t; op: H5D_operator_t; operator_data: Pointer): herr_t; cdecl;
    TH5Dvlen_reclaim = function(type_id: hid_t; space_id: hid_t; plist_id: hid_t; buf: Pointer): herr_t; cdecl;
    TH5Dvlen_get_buf_size = function(dataset_id: hid_t; type_id: hid_t; space_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Dfill = function(fill: Pointer; fill_type: hid_t; buf: Pointer; buf_type: hid_t; space: hid_t): herr_t; cdecl;
    TH5Dset_extent = function(dset_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Dflush = function(dset_id: hid_t): herr_t; cdecl;
    TH5Drefresh = function(dset_id: hid_t): herr_t; cdecl;
    TH5Dscatter = function(op: H5D_scatter_func_t; op_data: Pointer; type_id: hid_t; dst_space_id: hid_t; dst_buf: Pointer): herr_t; cdecl;
    TH5Dgather = function(src_space_id: hid_t; src_buf: Pointer; type_id: hid_t; dst_buf_size: size_t; dst_buf: Pointer; op: H5D_gather_func_t; op_data: Pointer): herr_t; cdecl;
    TH5Ddebug = function(dset_id: hid_t): herr_t; cdecl;
    TH5Dformat_convert = function(dset_id: hid_t): herr_t; cdecl;
    TH5Dget_chunk_index_type = function(did: hid_t; idx_type: PH5D_chunk_index_t): herr_t; cdecl;
    TH5Eregister_class = function(cls_name: PAnsiChar; lib_name: PAnsiChar; version: PAnsiChar): hid_t; cdecl;
    TH5Eunregister_class = function(class_id: hid_t): herr_t; cdecl;
    TH5Eclose_msg = function(err_id: hid_t): herr_t; cdecl;
    TH5Ecreate_msg = function(cls: hid_t; msg_type: H5E_type_t; msg: PAnsiChar): hid_t; cdecl;
    TH5Ecreate_stack = function: hid_t; cdecl;
    TH5Eget_current_stack = function: hid_t; cdecl;
    TH5Eclose_stack = function(stack_id: hid_t): herr_t; cdecl;
    TH5Eget_class_name = function(class_id: hid_t; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Eset_current_stack = function(err_stack_id: hid_t): herr_t; cdecl;
    // TH5Epush2 = function(err_stack: hid_t; file_: PAnsiChar; func: PAnsiChar; line: Cardinal; cls_id: hid_t; maj_id: hid_t; min_id: hid_t; msg: PAnsiChar): herr_t; cdecl; varargs;
    TH5Epop = function(err_stack: hid_t; count: size_t): herr_t; cdecl;
    TH5Eprint2 = function(err_stack: hid_t; stream: PFILE): herr_t; cdecl;
    TH5Ewalk2 = function(err_stack: hid_t; direction: H5E_direction_t; func: H5E_walk2_t; client_data: Pointer): herr_t; cdecl;
    TH5Eget_auto2 = function(estack_id: hid_t; func: PH5E_auto2_t; client_data: PPointer): herr_t; cdecl;
    TH5Eset_auto2 = function(estack_id: hid_t; func: H5E_auto2_t; client_data: Pointer): herr_t; cdecl;
    TH5Eclear2 = function(err_stack: hid_t): herr_t; cdecl;
    TH5Eauto_is_v2 = function(err_stack: hid_t; is_stack: PCardinal): herr_t; cdecl;
    TH5Eget_msg = function(msg_id: hid_t; typ: PH5E_type_t; msg: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Eget_num = function(error_stack_id: hid_t): ssize_t; cdecl;
    TH5Screate = function(typ: H5S_class_t): hid_t; cdecl;
    TH5Screate_simple = function(rank: Integer; dims: Phsize_t; maxdims: Phsize_t): hid_t; cdecl;
    TH5Sset_extent_simple = function(space_id: hid_t; rank: Integer; dims: Phsize_t; max: Phsize_t): herr_t; cdecl;
    TH5Scopy = function(space_id: hid_t): hid_t; cdecl;
    TH5Sclose = function(space_id: hid_t): herr_t; cdecl;
    TH5Sencode = function(obj_id: hid_t; buf: Pointer; nalloc: Psize_t): herr_t; cdecl;
    TH5Sdecode = function(buf: Pointer): hid_t; cdecl;
    TH5Sget_simple_extent_npoints = function(space_id: hid_t): hssize_t; cdecl;
    TH5Sget_simple_extent_ndims = function(space_id: hid_t): Integer; cdecl;
    TH5Sget_simple_extent_dims = function(space_id: hid_t; dims: Phsize_t; maxdims: Phsize_t): Integer; cdecl;
    TH5Sis_simple = function(space_id: hid_t): htri_t; cdecl;
    TH5Sget_select_npoints = function(spaceid: hid_t): hssize_t; cdecl;
    TH5Sselect_hyperslab = function(space_id: hid_t; op: H5S_seloper_t; start: Phsize_t; _stride: Phsize_t; count: Phsize_t; _block: Phsize_t): herr_t; cdecl;
    TH5Sselect_elements = function(space_id: hid_t; op: H5S_seloper_t; num_elem: size_t; coord: Phsize_t): herr_t; cdecl;
    TH5Sget_simple_extent_type = function(space_id: hid_t): H5S_class_t; cdecl;
    TH5Sset_extent_none = function(space_id: hid_t): herr_t; cdecl;
    TH5Sextent_copy = function(dst_id: hid_t; src_id: hid_t): herr_t; cdecl;
    TH5Sextent_equal = function(sid1: hid_t; sid2: hid_t): htri_t; cdecl;
    TH5Sselect_all = function(spaceid: hid_t): herr_t; cdecl;
    TH5Sselect_none = function(spaceid: hid_t): herr_t; cdecl;
    TH5Soffset_simple = function(space_id: hid_t; offset: Phssize_t): herr_t; cdecl;
    TH5Sselect_valid = function(spaceid: hid_t): htri_t; cdecl;
    TH5Sis_regular_hyperslab = function(spaceid: hid_t): htri_t; cdecl;
    TH5Sget_regular_hyperslab = function(spaceid: hid_t; start: Phsize_t; stride: Phsize_t; count: Phsize_t; block: Phsize_t): htri_t; cdecl;
    TH5Sget_select_hyper_nblocks = function(spaceid: hid_t): hssize_t; cdecl;
    TH5Sget_select_elem_npoints = function(spaceid: hid_t): hssize_t; cdecl;
    TH5Sget_select_hyper_blocklist = function(spaceid: hid_t; startblock: hsize_t; numblocks: hsize_t; buf: Phsize_t): herr_t; cdecl;
    TH5Sget_select_elem_pointlist = function(spaceid: hid_t; startpoint: hsize_t; numpoints: hsize_t; buf: Phsize_t): herr_t; cdecl;
    TH5Sget_select_bounds = function(spaceid: hid_t; start: Phsize_t; end_: Phsize_t): herr_t; cdecl;
    TH5Sget_select_type = function(spaceid: hid_t): H5S_sel_type; cdecl;
    TH5Lmove = function(src_loc: hid_t; src_name: PAnsiChar; dst_loc: hid_t; dst_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lcopy = function(src_loc: hid_t; src_name: PAnsiChar; dst_loc: hid_t; dst_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lcreate_hard = function(cur_loc: hid_t; cur_name: PAnsiChar; dst_loc: hid_t; dst_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lcreate_soft = function(link_target: PAnsiChar; link_loc_id: hid_t; link_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Ldelete = function(loc_id: hid_t; name: PAnsiChar; lapl_id: hid_t): herr_t; cdecl;
    TH5Ldelete_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lget_val = function(loc_id: hid_t; name: PAnsiChar; buf: Pointer; size: size_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lget_val_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; buf: Pointer; size: size_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lexists = function(loc_id: hid_t; name: PAnsiChar; lapl_id: hid_t): htri_t; cdecl;
    TH5Lget_info = function(loc_id: hid_t; name: PAnsiChar; linfo: PH5L_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lget_info_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; linfo: PH5L_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lget_name_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; name: PAnsiChar; size: size_t; lapl_id: hid_t): ssize_t; cdecl;
    TH5Literate = function(grp_id: hid_t; idx_type: H5_index_t; order: H5_iter_order_t; idx: Phsize_t; op: H5L_iterate_t; op_data: Pointer): herr_t; cdecl;
    TH5Literate_by_name = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; idx: Phsize_t; op: H5L_iterate_t; op_data: Pointer; lapl_id: hid_t): herr_t; cdecl;
    TH5Lvisit = function(grp_id: hid_t; idx_type: H5_index_t; order: H5_iter_order_t; op: H5L_iterate_t; op_data: Pointer): herr_t; cdecl;
    TH5Lvisit_by_name = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; op: H5L_iterate_t; op_data: Pointer; lapl_id: hid_t): herr_t; cdecl;
    TH5Lcreate_ud = function(link_loc_id: hid_t; link_name: PAnsiChar; link_type: H5L_type_t; udata: Pointer; udata_size: size_t; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Lregister = function(cls: PH5L_class_t): herr_t; cdecl;
    TH5Lunregister = function(id: H5L_type_t): herr_t; cdecl;
    TH5Lis_registered = function(id: H5L_type_t): htri_t; cdecl;
    TH5Lunpack_elink_val = function(ext_linkval: Pointer; link_size: size_t; flags: PCardinal; filename: PPAnsiChar; obj_path: PPAnsiChar): herr_t; cdecl;
    TH5Lcreate_external = function(file_name: PAnsiChar; obj_name: PAnsiChar; link_loc_id: hid_t; link_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Oopen = function(loc_id: hid_t; name: PAnsiChar; lapl_id: hid_t): hid_t; cdecl;
    TH5Oopen_by_addr = function(loc_id: hid_t; addr: haddr_t): hid_t; cdecl;
    TH5Oopen_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; lapl_id: hid_t): hid_t; cdecl;
    TH5Oexists_by_name = function(loc_id: hid_t; name: PAnsiChar; lapl_id: hid_t): htri_t; cdecl;
    TH5Oget_info = function(loc_id: hid_t; oinfo: PH5O_info_t): herr_t; cdecl;
    TH5Oget_info_by_name = function(loc_id: hid_t; name: PAnsiChar; oinfo: PH5O_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Oget_info_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; oinfo: PH5O_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Olink = function(obj_id: hid_t; new_loc_id: hid_t; new_name: PAnsiChar; lcpl_id: hid_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Oincr_refcount = function(object_id: hid_t): herr_t; cdecl;
    TH5Odecr_refcount = function(object_id: hid_t): herr_t; cdecl;
    TH5Ocopy = function(src_loc_id: hid_t; src_name: PAnsiChar; dst_loc_id: hid_t; dst_name: PAnsiChar; ocpypl_id: hid_t; lcpl_id: hid_t): herr_t; cdecl;
    TH5Oset_comment = function(obj_id: hid_t; comment: PAnsiChar): herr_t; cdecl;
    TH5Oset_comment_by_name = function(loc_id: hid_t; name: PAnsiChar; comment: PAnsiChar; lapl_id: hid_t): herr_t; cdecl;
    TH5Oget_comment = function(obj_id: hid_t; comment: PAnsiChar; bufsize: size_t): ssize_t; cdecl;
    TH5Oget_comment_by_name = function(loc_id: hid_t; name: PAnsiChar; comment: PAnsiChar; bufsize: size_t; lapl_id: hid_t): ssize_t; cdecl;
    TH5Ovisit = function(obj_id: hid_t; idx_type: H5_index_t; order: H5_iter_order_t; op: H5O_iterate_t; op_data: Pointer): herr_t; cdecl;
    TH5Ovisit_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; op: H5O_iterate_t; op_data: Pointer; lapl_id: hid_t): herr_t; cdecl;
    TH5Oclose = function(object_id: hid_t): herr_t; cdecl;
    TH5Oflush = function(obj_id: hid_t): herr_t; cdecl;
    TH5Orefresh = function(oid: hid_t): herr_t; cdecl;
    TH5Odisable_mdc_flushes = function(object_id: hid_t): herr_t; cdecl;
    TH5Oenable_mdc_flushes = function(object_id: hid_t): herr_t; cdecl;
    TH5Oare_mdc_flushes_disabled = function(object_id: hid_t; are_disabled: Phbool_t): herr_t; cdecl;
    TH5Fis_hdf5 = function(filename: PAnsiChar): htri_t; cdecl;
    TH5Fcreate = function(filename: PAnsiChar; flags: Cardinal; create_plist: hid_t; access_plist: hid_t): hid_t; cdecl;
    TH5Fopen = function(filename: PAnsiChar; flags: Cardinal; access_plist: hid_t): hid_t; cdecl;
    TH5Freopen = function(file_id: hid_t): hid_t; cdecl;
    TH5Fflush = function(object_id: hid_t; scope: H5F_scope_t): herr_t; cdecl;
    TH5Fclose = function(file_id: hid_t): herr_t; cdecl;
    TH5Fget_create_plist = function(file_id: hid_t): hid_t; cdecl;
    TH5Fget_access_plist = function(file_id: hid_t): hid_t; cdecl;
    TH5Fget_intent = function(file_id: hid_t; intent: PCardinal): herr_t; cdecl;
    TH5Fget_obj_count = function(file_id: hid_t; types: Cardinal): ssize_t; cdecl;
    TH5Fget_obj_ids = function(file_id: hid_t; types: Cardinal; max_objs: size_t; obj_id_list: Phid_t): ssize_t; cdecl;
    TH5Fget_vfd_handle = function(file_id: hid_t; fapl: hid_t; file_handle: PPointer): herr_t; cdecl;
    TH5Fmount = function(loc: hid_t; name: PAnsiChar; child: hid_t; plist: hid_t): herr_t; cdecl;
    TH5Funmount = function(loc: hid_t; name: PAnsiChar): herr_t; cdecl;
    TH5Fget_freespace = function(file_id: hid_t): hssize_t; cdecl;
    TH5Fget_filesize = function(file_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Fget_file_image = function(file_id: hid_t; buf_ptr: Pointer; buf_len: size_t): ssize_t; cdecl;
    TH5Fget_mdc_config = function(file_id: hid_t; config_ptr: PH5AC_cache_config_t): herr_t; cdecl;
    TH5Fset_mdc_config = function(file_id: hid_t; config_ptr: PH5AC_cache_config_t): herr_t; cdecl;
    TH5Fget_mdc_hit_rate = function(file_id: hid_t; hit_rate_ptr: PDouble): herr_t; cdecl;
    TH5Fget_mdc_size = function(file_id: hid_t; max_size_ptr: Psize_t; min_clean_size_ptr: Psize_t; cur_size_ptr: Psize_t; cur_num_entries_ptr: PInteger): herr_t; cdecl;
    TH5Freset_mdc_hit_rate_stats = function(file_id: hid_t): herr_t; cdecl;
    TH5Fget_name = function(obj_id: hid_t; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Fget_info2 = function(obj_id: hid_t; finfo: PH5F_info2_t): herr_t; cdecl;
    TH5Fget_metadata_read_retry_info = function(file_id: hid_t; info: PH5F_retry_info_t): herr_t; cdecl;
    TH5Fstart_swmr_write = function(file_id: hid_t): herr_t; cdecl;
    TH5Fget_free_sections = function(file_id: hid_t; typ: H5F_mem_t; nsects: size_t; sect_info: PH5F_sect_info_t): ssize_t; cdecl;
    TH5Fclear_elink_file_cache = function(file_id: hid_t): herr_t; cdecl;
    TH5Fstart_mdc_logging = function(file_id: hid_t): herr_t; cdecl;
    TH5Fstop_mdc_logging = function(file_id: hid_t): herr_t; cdecl;
    TH5Fget_mdc_logging_status = function(file_id: hid_t; is_enabled: Phbool_t; is_currently_logging: Phbool_t): herr_t; cdecl;
    TH5Fformat_convert = function(fid: hid_t): herr_t; cdecl;
    TH5Acreate2 = function(loc_id: hid_t; attr_name: PAnsiChar; type_id: hid_t; space_id: hid_t; acpl_id: hid_t; aapl_id: hid_t): hid_t; cdecl;
    TH5Acreate_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; attr_name: PAnsiChar; type_id: hid_t; space_id: hid_t; acpl_id: hid_t; aapl_id: hid_t; lapl_id: hid_t): hid_t; cdecl;
    TH5Aopen = function(obj_id: hid_t; attr_name: PAnsiChar; aapl_id: hid_t): hid_t; cdecl;
    TH5Aopen_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; attr_name: PAnsiChar; aapl_id: hid_t; lapl_id: hid_t): hid_t; cdecl;
    TH5Aopen_by_idx = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; aapl_id: hid_t; lapl_id: hid_t): hid_t; cdecl;
    TH5Awrite = function(attr_id: hid_t; type_id: hid_t; buf: Pointer): herr_t; cdecl;
    TH5Aread = function(attr_id: hid_t; type_id: hid_t; buf: Pointer): herr_t; cdecl;
    TH5Aclose = function(attr_id: hid_t): herr_t; cdecl;
    TH5Aget_space = function(attr_id: hid_t): hid_t; cdecl;
    TH5Aget_type = function(attr_id: hid_t): hid_t; cdecl;
    TH5Aget_create_plist = function(attr_id: hid_t): hid_t; cdecl;
    TH5Aget_name = function(attr_id: hid_t; buf_size: size_t; buf: PAnsiChar): ssize_t; cdecl;
    TH5Aget_name_by_idx = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; name: PAnsiChar; size: size_t; lapl_id: hid_t): ssize_t; cdecl;
    TH5Aget_storage_size = function(attr_id: hid_t): hsize_t; cdecl;
    TH5Aget_info = function(attr_id: hid_t; ainfo: PH5A_info_t): herr_t; cdecl;
    TH5Aget_info_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; attr_name: PAnsiChar; ainfo: PH5A_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Aget_info_by_idx = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; ainfo: PH5A_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Arename = function(loc_id: hid_t; old_name: PAnsiChar; new_name: PAnsiChar): herr_t; cdecl;
    TH5Arename_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; old_attr_name: PAnsiChar; new_attr_name: PAnsiChar; lapl_id: hid_t): herr_t; cdecl;
    TH5Aiterate2 = function(loc_id: hid_t; idx_type: H5_index_t; order: H5_iter_order_t; idx: Phsize_t; op: H5A_operator2_t; op_data: Pointer): herr_t; cdecl;
    TH5Aiterate_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; idx: Phsize_t; op: H5A_operator2_t; op_data: Pointer; lapd_id: hid_t): herr_t; cdecl;
    TH5Adelete = function(loc_id: hid_t; name: PAnsiChar): herr_t; cdecl;
    TH5Adelete_by_name = function(loc_id: hid_t; obj_name: PAnsiChar; attr_name: PAnsiChar; lapl_id: hid_t): herr_t; cdecl;
    TH5Adelete_by_idx = function(loc_id: hid_t; obj_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Aexists = function(obj_id: hid_t; attr_name: PAnsiChar): htri_t; cdecl;
    TH5Aexists_by_name = function(obj_id: hid_t; obj_name: PAnsiChar; attr_name: PAnsiChar; lapl_id: hid_t): htri_t; cdecl;
    TH5FDregister = function(cls: PH5FD_class_t): hid_t; cdecl;
    TH5FDunregister = function(driver_id: hid_t): herr_t; cdecl;
    TH5FDopen = function(name: PAnsiChar; flags: Cardinal; fapl_id: hid_t; maxaddr: haddr_t): PH5FD_t; cdecl;
    TH5FDclose = function(file_: PH5FD_t): herr_t; cdecl;
    TH5FDcmp = function(f1: PH5FD_t; f2: PH5FD_t): Integer; cdecl;
    TH5FDquery = function(f: PH5FD_t; flags: PCardinal): Integer; cdecl;
    TH5FDalloc = function(file_: PH5FD_t; typ: H5FD_mem_t; dxpl_id: hid_t; size: hsize_t): haddr_t; cdecl;
    TH5FDfree = function(file_: PH5FD_t; typ: H5FD_mem_t; dxpl_id: hid_t; addr: haddr_t; size: hsize_t): herr_t; cdecl;
    TH5FDget_eoa = function(file_: PH5FD_t; typ: H5FD_mem_t): haddr_t; cdecl;
    TH5FDset_eoa = function(file_: PH5FD_t; typ: H5FD_mem_t; eoa: haddr_t): herr_t; cdecl;
    TH5FDget_eof = function(file_: PH5FD_t; typ: H5FD_mem_t): haddr_t; cdecl;
    TH5FDget_vfd_handle = function(file_: PH5FD_t; fapl: hid_t; file_handle: PPointer): herr_t; cdecl;
    TH5FDread = function(file_: PH5FD_t; typ: H5FD_mem_t; dxpl_id: hid_t; addr: haddr_t; size: size_t; buf: Pointer): herr_t; cdecl;
    TH5FDwrite = function(file_: PH5FD_t; typ: H5FD_mem_t; dxpl_id: hid_t; addr: haddr_t; size: size_t; buf: Pointer): herr_t; cdecl;
    TH5FDflush = function(file_: PH5FD_t; dxpl_id: hid_t; closing: Cardinal): herr_t; cdecl;
    TH5FDtruncate = function(file_: PH5FD_t; dxpl_id: hid_t; closing: hbool_t): herr_t; cdecl;
    TH5FDlock = function(file_: PH5FD_t; rw: hbool_t): herr_t; cdecl;
    TH5FDunlock = function(file_: PH5FD_t): herr_t; cdecl;
    TH5Gcreate2 = function(loc_id: hid_t; name: PAnsiChar; lcpl_id: hid_t; gcpl_id: hid_t; gapl_id: hid_t): hid_t; cdecl;
    TH5Gcreate_anon = function(loc_id: hid_t; gcpl_id: hid_t; gapl_id: hid_t): hid_t; cdecl;
    TH5Gopen2 = function(loc_id: hid_t; name: PAnsiChar; gapl_id: hid_t): hid_t; cdecl;
    TH5Gget_create_plist = function(group_id: hid_t): hid_t; cdecl;
    TH5Gget_info = function(loc_id: hid_t; ginfo: PH5G_info_t): herr_t; cdecl;
    TH5Gget_info_by_name = function(loc_id: hid_t; name: PAnsiChar; ginfo: PH5G_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Gget_info_by_idx = function(loc_id: hid_t; group_name: PAnsiChar; idx_type: H5_index_t; order: H5_iter_order_t; n: hsize_t; ginfo: PH5G_info_t; lapl_id: hid_t): herr_t; cdecl;
    TH5Gclose = function(group_id: hid_t): herr_t; cdecl;
    TH5Gflush = function(group_id: hid_t): herr_t; cdecl;
    TH5Grefresh = function(group_id: hid_t): herr_t; cdecl;
    TH5Rcreate = function(ref: Pointer; loc_id: hid_t; name: PAnsiChar; ref_type: H5R_type_t; space_id: hid_t): herr_t; cdecl;
    TH5Rdereference2 = function(obj_id: hid_t; oapl_id: hid_t; ref_type: H5R_type_t; ref: Pointer): hid_t; cdecl;
    TH5Rget_region = function(dataset: hid_t; ref_type: H5R_type_t; ref: Pointer): hid_t; cdecl;
    TH5Rget_obj_type2 = function(id: hid_t; ref_type: H5R_type_t; _ref: Pointer; obj_type: PH5O_type_t): herr_t; cdecl;
    TH5Rget_name = function(loc_id: hid_t; ref_type: H5R_type_t; ref: Pointer; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pcreate_class = function(parent: hid_t; name: PAnsiChar; cls_create: H5P_cls_create_func_t; create_data: Pointer; cls_copy: H5P_cls_copy_func_t; copy_data: Pointer; cls_close: H5P_cls_close_func_t; close_data: Pointer): hid_t; cdecl;
    TH5Pget_class_name = function(pclass_id: hid_t): PAnsiChar; cdecl;
    TH5Pcreate = function(cls_id: hid_t): hid_t; cdecl;
    TH5Pregister2 = function(cls_id: hid_t; name: PAnsiChar; size: size_t; def_value: Pointer; prp_create: H5P_prp_create_func_t; prp_set: H5P_prp_set_func_t; prp_get: H5P_prp_get_func_t; prp_del: H5P_prp_delete_func_t; prp_copy: H5P_prp_copy_func_t; prp_cmp: H5P_prp_compare_func_t; prp_close: H5P_prp_close_func_t): herr_t; cdecl;
    TH5Pinsert2 = function(plist_id: hid_t; name: PAnsiChar; size: size_t; value: Pointer; prp_set: H5P_prp_set_func_t; prp_get: H5P_prp_get_func_t; prp_delete: H5P_prp_delete_func_t; prp_copy: H5P_prp_copy_func_t; prp_cmp: H5P_prp_compare_func_t; prp_close: H5P_prp_close_func_t): herr_t; cdecl;
    TH5Pset = function(plist_id: hid_t; name: PAnsiChar; value: Pointer): herr_t; cdecl;
    TH5Pexist = function(plist_id: hid_t; name: PAnsiChar): htri_t; cdecl;
    TH5Pencode = function(plist_id: hid_t; buf: Pointer; nalloc: Psize_t): herr_t; cdecl;
    TH5Pdecode = function(buf: Pointer): hid_t; cdecl;
    TH5Pget_size = function(id: hid_t; name: PAnsiChar; size: Psize_t): herr_t; cdecl;
    TH5Pget_nprops = function(id: hid_t; nprops: Psize_t): herr_t; cdecl;
    TH5Pget_class = function(plist_id: hid_t): hid_t; cdecl;
    TH5Pget_class_parent = function(pclass_id: hid_t): hid_t; cdecl;
    TH5Pget = function(plist_id: hid_t; name: PAnsiChar; value: Pointer): herr_t; cdecl;
    TH5Pequal = function(id1: hid_t; id2: hid_t): htri_t; cdecl;
    TH5Pisa_class = function(plist_id: hid_t; pclass_id: hid_t): htri_t; cdecl;
    TH5Piterate = function(id: hid_t; idx: PInteger; iter_func: H5P_iterate_t; iter_data: Pointer): Integer; cdecl;
    TH5Pcopy_prop = function(dst_id: hid_t; src_id: hid_t; name: PAnsiChar): herr_t; cdecl;
    TH5Premove = function(plist_id: hid_t; name: PAnsiChar): herr_t; cdecl;
    TH5Punregister = function(pclass_id: hid_t; name: PAnsiChar): herr_t; cdecl;
    TH5Pclose_class = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pclose = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pcopy = function(plist_id: hid_t): hid_t; cdecl;
    TH5Pset_attr_phase_change = function(plist_id: hid_t; max_compact: Cardinal; min_dense: Cardinal): herr_t; cdecl;
    TH5Pget_attr_phase_change = function(plist_id: hid_t; max_compact: PCardinal; min_dense: PCardinal): herr_t; cdecl;
    TH5Pset_attr_creation_order = function(plist_id: hid_t; crt_order_flags: Cardinal): herr_t; cdecl;
    TH5Pget_attr_creation_order = function(plist_id: hid_t; crt_order_flags: PCardinal): herr_t; cdecl;
    TH5Pset_obj_track_times = function(plist_id: hid_t; track_times: hbool_t): herr_t; cdecl;
    TH5Pget_obj_track_times = function(plist_id: hid_t; track_times: Phbool_t): herr_t; cdecl;
    TH5Pmodify_filter = function(plist_id: hid_t; filter: H5Z_filter_t; flags: Cardinal; cd_nelmts: size_t; cd_values: PCardinal): herr_t; cdecl;
    TH5Pset_filter = function(plist_id: hid_t; filter: H5Z_filter_t; flags: Cardinal; cd_nelmts: size_t; c_values: PCardinal): herr_t; cdecl;
    TH5Pget_nfilters = function(plist_id: hid_t): Integer; cdecl;
    TH5Pget_filter2 = function(plist_id: hid_t; filter: Cardinal; flags: PCardinal; cd_nelmts: Psize_t; cd_values: PCardinal; namelen: size_t; name: PAnsiChar; filter_config: PCardinal): H5Z_filter_t; cdecl;
    TH5Pget_filter_by_id2 = function(plist_id: hid_t; id: H5Z_filter_t; flags: PCardinal; cd_nelmts: Psize_t; cd_values: PCardinal; namelen: size_t; name: PAnsiChar; filter_config: PCardinal): herr_t; cdecl;
    TH5Pall_filters_avail = function(plist_id: hid_t): htri_t; cdecl;
    TH5Premove_filter = function(plist_id: hid_t; filter: H5Z_filter_t): herr_t; cdecl;
    TH5Pset_deflate = function(plist_id: hid_t; aggression: Cardinal): herr_t; cdecl;
    TH5Pset_fletcher32 = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pset_userblock = function(plist_id: hid_t; size: hsize_t): herr_t; cdecl;
    TH5Pget_userblock = function(plist_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Pset_sizes = function(plist_id: hid_t; sizeof_addr: size_t; sizeof_size: size_t): herr_t; cdecl;
    TH5Pget_sizes = function(plist_id: hid_t; sizeof_addr: Psize_t; sizeof_size: Psize_t): herr_t; cdecl;
    TH5Pset_sym_k = function(plist_id: hid_t; ik: Cardinal; lk: Cardinal): herr_t; cdecl;
    TH5Pget_sym_k = function(plist_id: hid_t; ik: PCardinal; lk: PCardinal): herr_t; cdecl;
    TH5Pset_istore_k = function(plist_id: hid_t; ik: Cardinal): herr_t; cdecl;
    TH5Pget_istore_k = function(plist_id: hid_t; ik: PCardinal): herr_t; cdecl;
    TH5Pset_shared_mesg_nindexes = function(plist_id: hid_t; nindexes: Cardinal): herr_t; cdecl;
    TH5Pget_shared_mesg_nindexes = function(plist_id: hid_t; nindexes: PCardinal): herr_t; cdecl;
    TH5Pset_shared_mesg_index = function(plist_id: hid_t; index_num: Cardinal; mesg_type_flags: Cardinal; min_mesg_size: Cardinal): herr_t; cdecl;
    TH5Pget_shared_mesg_index = function(plist_id: hid_t; index_num: Cardinal; mesg_type_flags: PCardinal; min_mesg_size: PCardinal): herr_t; cdecl;
    TH5Pset_shared_mesg_phase_change = function(plist_id: hid_t; max_list: Cardinal; min_btree: Cardinal): herr_t; cdecl;
    TH5Pget_shared_mesg_phase_change = function(plist_id: hid_t; max_list: PCardinal; min_btree: PCardinal): herr_t; cdecl;
    TH5Pset_file_space = function(plist_id: hid_t; strategy: H5F_file_space_type_t; threshold: hsize_t): herr_t; cdecl;
    TH5Pget_file_space = function(plist_id: hid_t; strategy: PH5F_file_space_type_t; threshold: Phsize_t): herr_t; cdecl;
    TH5Pset_alignment = function(fapl_id: hid_t; threshold: hsize_t; alignment: hsize_t): herr_t; cdecl;
    TH5Pget_alignment = function(fapl_id: hid_t; threshold: Phsize_t; alignment: Phsize_t): herr_t; cdecl;
    TH5Pset_driver = function(plist_id: hid_t; driver_id: hid_t; driver_info: Pointer): herr_t; cdecl;
    TH5Pget_driver = function(plist_id: hid_t): hid_t; cdecl;
    TH5Pget_driver_info = function(plist_id: hid_t): Pointer; cdecl;
    TH5Pset_family_offset = function(fapl_id: hid_t; offset: hsize_t): herr_t; cdecl;
    TH5Pget_family_offset = function(fapl_id: hid_t; offset: Phsize_t): herr_t; cdecl;
    TH5Pset_multi_type = function(fapl_id: hid_t; typ: H5FD_mem_t): herr_t; cdecl;
    TH5Pget_multi_type = function(fapl_id: hid_t; typ: PH5FD_mem_t): herr_t; cdecl;
    TH5Pset_cache = function(plist_id: hid_t; mdc_nelmts: Integer; rdcc_nslots: size_t; rdcc_nbytes: size_t; rdcc_w0: Double): herr_t; cdecl;
    TH5Pget_cache = function(plist_id: hid_t; mdc_nelmts: PInteger; rdcc_nslots: Psize_t; rdcc_nbytes: Psize_t; rdcc_w0: PDouble): herr_t; cdecl;
    TH5Pset_mdc_config = function(plist_id: hid_t; config_ptr: PH5AC_cache_config_t): herr_t; cdecl;
    TH5Pget_mdc_config = function(plist_id: hid_t; config_ptr: PH5AC_cache_config_t): herr_t; cdecl;
    TH5Pset_gc_references = function(fapl_id: hid_t; gc_ref: Cardinal): herr_t; cdecl;
    TH5Pget_gc_references = function(fapl_id: hid_t; gc_ref: PCardinal): herr_t; cdecl;
    TH5Pset_fclose_degree = function(fapl_id: hid_t; degree: H5F_close_degree_t): herr_t; cdecl;
    TH5Pget_fclose_degree = function(fapl_id: hid_t; degree: PH5F_close_degree_t): herr_t; cdecl;
    TH5Pset_meta_block_size = function(fapl_id: hid_t; size: hsize_t): herr_t; cdecl;
    TH5Pget_meta_block_size = function(fapl_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Pset_sieve_buf_size = function(fapl_id: hid_t; size: size_t): herr_t; cdecl;
    TH5Pget_sieve_buf_size = function(fapl_id: hid_t; size: Psize_t): herr_t; cdecl;
    TH5Pset_small_data_block_size = function(fapl_id: hid_t; size: hsize_t): herr_t; cdecl;
    TH5Pget_small_data_block_size = function(fapl_id: hid_t; size: Phsize_t): herr_t; cdecl;
    TH5Pset_libver_bounds = function(plist_id: hid_t; low: H5F_libver_t; high: H5F_libver_t): herr_t; cdecl;
    TH5Pget_libver_bounds = function(plist_id: hid_t; low: PH5F_libver_t; high: PH5F_libver_t): herr_t; cdecl;
    TH5Pset_elink_file_cache_size = function(plist_id: hid_t; efc_size: Cardinal): herr_t; cdecl;
    TH5Pget_elink_file_cache_size = function(plist_id: hid_t; efc_size: PCardinal): herr_t; cdecl;
    TH5Pset_file_image = function(fapl_id: hid_t; buf_ptr: Pointer; buf_len: size_t): herr_t; cdecl;
    TH5Pget_file_image = function(fapl_id: hid_t; buf_ptr_ptr: PPointer; buf_len_ptr: Psize_t): herr_t; cdecl;
    TH5Pset_file_image_callbacks = function(fapl_id: hid_t; callbacks_ptr: PH5FD_file_image_callbacks_t): herr_t; cdecl;
    TH5Pget_file_image_callbacks = function(fapl_id: hid_t; callbacks_ptr: PH5FD_file_image_callbacks_t): herr_t; cdecl;
    TH5Pset_core_write_tracking = function(fapl_id: hid_t; is_enabled: hbool_t; page_size: size_t): herr_t; cdecl;
    TH5Pget_core_write_tracking = function(fapl_id: hid_t; is_enabled: Phbool_t; page_size: Psize_t): herr_t; cdecl;
    TH5Pset_metadata_read_attempts = function(plist_id: hid_t; attempts: Cardinal): herr_t; cdecl;
    TH5Pget_metadata_read_attempts = function(plist_id: hid_t; attempts: PCardinal): herr_t; cdecl;
    TH5Pset_object_flush_cb = function(plist_id: hid_t; func: H5F_flush_cb_t; udata: Pointer): herr_t; cdecl;
    TH5Pget_object_flush_cb = function(plist_id: hid_t; func: PH5F_flush_cb_t; udata: PPointer): herr_t; cdecl;
    TH5Pset_mdc_log_options = function(plist_id: hid_t; is_enabled: hbool_t; location: PAnsiChar; start_on_access: hbool_t): herr_t; cdecl;
    TH5Pget_mdc_log_options = function(plist_id: hid_t; is_enabled: Phbool_t; location: PAnsiChar; location_size: Psize_t; start_on_access: Phbool_t): herr_t; cdecl;
    TH5Pset_layout = function(plist_id: hid_t; layout: H5D_layout_t): herr_t; cdecl;
    TH5Pget_layout = function(plist_id: hid_t): H5D_layout_t; cdecl;
    TH5Pset_chunk = function(plist_id: hid_t; ndims: Integer; dim: Phsize_t): herr_t; cdecl;
    TH5Pget_chunk = function(plist_id: hid_t; max_ndims: Integer; dim: Phsize_t): Integer; cdecl;
    TH5Pset_virtual = function(dcpl_id: hid_t; vspace_id: hid_t; src_file_name: PAnsiChar; src_dset_name: PAnsiChar; src_space_id: hid_t): herr_t; cdecl;
    TH5Pget_virtual_count = function(dcpl_id: hid_t; count: Psize_t): herr_t; cdecl;
    TH5Pget_virtual_vspace = function(dcpl_id: hid_t; index: size_t): hid_t; cdecl;
    TH5Pget_virtual_srcspace = function(dcpl_id: hid_t; index: size_t): hid_t; cdecl;
    TH5Pget_virtual_filename = function(dcpl_id: hid_t; index: size_t; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pget_virtual_dsetname = function(dcpl_id: hid_t; index: size_t; name: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pset_external = function(plist_id: hid_t; name: PAnsiChar; offset: off_t; size: hsize_t): herr_t; cdecl;
    TH5Pset_chunk_opts = function(plist_id: hid_t; opts: Cardinal): herr_t; cdecl;
    TH5Pget_chunk_opts = function(plist_id: hid_t; opts: PCardinal): herr_t; cdecl;
    TH5Pget_external_count = function(plist_id: hid_t): Integer; cdecl;
    TH5Pget_external = function(plist_id: hid_t; idx: Cardinal; name_size: size_t; name: PAnsiChar; offset: Poff_t; size: Phsize_t): herr_t; cdecl;
    TH5Pset_szip = function(plist_id: hid_t; options_mask: Cardinal; pixels_per_block: Cardinal): herr_t; cdecl;
    TH5Pset_shuffle = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pset_nbit = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pset_scaleoffset = function(plist_id: hid_t; scale_type: H5Z_SO_scale_type_t; scale_factor: Integer): herr_t; cdecl;
    TH5Pset_fill_value = function(plist_id: hid_t; type_id: hid_t; value: Pointer): herr_t; cdecl;
    TH5Pget_fill_value = function(plist_id: hid_t; type_id: hid_t; value: Pointer): herr_t; cdecl;
    TH5Pfill_value_defined = function(plist: hid_t; status: PH5D_fill_value_t): herr_t; cdecl;
    TH5Pset_alloc_time = function(plist_id: hid_t; alloc_time: H5D_alloc_time_t): herr_t; cdecl;
    TH5Pget_alloc_time = function(plist_id: hid_t; alloc_time: PH5D_alloc_time_t): herr_t; cdecl;
    TH5Pset_fill_time = function(plist_id: hid_t; fill_time: H5D_fill_time_t): herr_t; cdecl;
    TH5Pget_fill_time = function(plist_id: hid_t; fill_time: PH5D_fill_time_t): herr_t; cdecl;
    TH5Pset_chunk_cache = function(dapl_id: hid_t; rdcc_nslots: size_t; rdcc_nbytes: size_t; rdcc_w0: Double): herr_t; cdecl;
    TH5Pget_chunk_cache = function(dapl_id: hid_t; rdcc_nslots: Psize_t; rdcc_nbytes: Psize_t; rdcc_w0: PDouble): herr_t; cdecl;
    TH5Pset_virtual_view = function(plist_id: hid_t; view: H5D_vds_view_t): herr_t; cdecl;
    TH5Pget_virtual_view = function(plist_id: hid_t; view: PH5D_vds_view_t): herr_t; cdecl;
    TH5Pset_virtual_printf_gap = function(plist_id: hid_t; gap_size: hsize_t): herr_t; cdecl;
    TH5Pget_virtual_printf_gap = function(plist_id: hid_t; gap_size: Phsize_t): herr_t; cdecl;
    TH5Pset_append_flush = function(plist_id: hid_t; ndims: Cardinal; boundary: Phsize_t; func: H5D_append_cb_t; udata: Pointer): herr_t; cdecl;
    TH5Pget_append_flush = function(plist_id: hid_t; dims: Cardinal; boundary: Phsize_t; func: PH5D_append_cb_t; udata: PPointer): herr_t; cdecl;
    TH5Pset_efile_prefix = function(dapl_id: hid_t; prefix: PAnsiChar): herr_t; cdecl;
    TH5Pget_efile_prefix = function(dapl_id: hid_t; prefix: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pset_data_transform = function(plist_id: hid_t; expression: PAnsiChar): herr_t; cdecl;
    TH5Pget_data_transform = function(plist_id: hid_t; expression: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pset_buffer = function(plist_id: hid_t; size: size_t; tconv: Pointer; bkg: Pointer): herr_t; cdecl;
    TH5Pget_buffer = function(plist_id: hid_t; tconv: PPointer; bkg: PPointer): size_t; cdecl;
    TH5Pset_preserve = function(plist_id: hid_t; status: hbool_t): herr_t; cdecl;
    TH5Pget_preserve = function(plist_id: hid_t): Integer; cdecl;
    TH5Pset_edc_check = function(plist_id: hid_t; check: H5Z_EDC_t): herr_t; cdecl;
    TH5Pget_edc_check = function(plist_id: hid_t): H5Z_EDC_t; cdecl;
    TH5Pset_filter_callback = function(plist_id: hid_t; func: H5Z_filter_func_t; op_data: Pointer): herr_t; cdecl;
    TH5Pset_btree_ratios = function(plist_id: hid_t; left: Double; middle: Double; right: Double): herr_t; cdecl;
    TH5Pget_btree_ratios = function(plist_id: hid_t; left: PDouble; middle: PDouble; right: PDouble): herr_t; cdecl;
    TH5Pset_vlen_mem_manager = function(plist_id: hid_t; alloc_func: H5MM_allocate_t; alloc_info: Pointer; free_func: H5MM_free_t; free_info: Pointer): herr_t; cdecl;
    TH5Pget_vlen_mem_manager = function(plist_id: hid_t; alloc_func: PH5MM_allocate_t; alloc_info: PPointer; free_func: PH5MM_free_t; free_info: PPointer): herr_t; cdecl;
    TH5Pset_hyper_vector_size = function(fapl_id: hid_t; size: size_t): herr_t; cdecl;
    TH5Pget_hyper_vector_size = function(fapl_id: hid_t; size: Psize_t): herr_t; cdecl;
    TH5Pset_type_conv_cb = function(dxpl_id: hid_t; op: H5T_conv_except_func_t; operate_data: Pointer): herr_t; cdecl;
    TH5Pget_type_conv_cb = function(dxpl_id: hid_t; op: PH5T_conv_except_func_t; operate_data: PPointer): herr_t; cdecl;
    TH5Pset_create_intermediate_group = function(plist_id: hid_t; crt_intmd: Cardinal): herr_t; cdecl;
    TH5Pget_create_intermediate_group = function(plist_id: hid_t; crt_intmd: PCardinal): herr_t; cdecl;
    TH5Pset_local_heap_size_hint = function(plist_id: hid_t; size_hint: size_t): herr_t; cdecl;
    TH5Pget_local_heap_size_hint = function(plist_id: hid_t; size_hint: Psize_t): herr_t; cdecl;
    TH5Pset_link_phase_change = function(plist_id: hid_t; max_compact: Cardinal; min_dense: Cardinal): herr_t; cdecl;
    TH5Pget_link_phase_change = function(plist_id: hid_t; max_compact: PCardinal; min_dense: PCardinal): herr_t; cdecl;
    TH5Pset_est_link_info = function(plist_id: hid_t; est_num_entries: Cardinal; est_name_len: Cardinal): herr_t; cdecl;
    TH5Pget_est_link_info = function(plist_id: hid_t; est_num_entries: PCardinal; est_name_len: PCardinal): herr_t; cdecl;
    TH5Pset_link_creation_order = function(plist_id: hid_t; crt_order_flags: Cardinal): herr_t; cdecl;
    TH5Pget_link_creation_order = function(plist_id: hid_t; crt_order_flags: PCardinal): herr_t; cdecl;
    TH5Pset_char_encoding = function(plist_id: hid_t; encoding: H5T_cset_t): herr_t; cdecl;
    TH5Pget_char_encoding = function(plist_id: hid_t; encoding: PH5T_cset_t): herr_t; cdecl;
    TH5Pset_nlinks = function(plist_id: hid_t; nlinks: size_t): herr_t; cdecl;
    TH5Pget_nlinks = function(plist_id: hid_t; nlinks: Psize_t): herr_t; cdecl;
    TH5Pset_elink_prefix = function(plist_id: hid_t; prefix: PAnsiChar): herr_t; cdecl;
    TH5Pget_elink_prefix = function(plist_id: hid_t; prefix: PAnsiChar; size: size_t): ssize_t; cdecl;
    TH5Pget_elink_fapl = function(lapl_id: hid_t): hid_t; cdecl;
    TH5Pset_elink_fapl = function(lapl_id: hid_t; fapl_id: hid_t): herr_t; cdecl;
    TH5Pset_elink_acc_flags = function(lapl_id: hid_t; flags: Cardinal): herr_t; cdecl;
    TH5Pget_elink_acc_flags = function(lapl_id: hid_t; flags: PCardinal): herr_t; cdecl;
    TH5Pset_elink_cb = function(lapl_id: hid_t; func: H5L_elink_traverse_t; op_data: Pointer): herr_t; cdecl;
    TH5Pget_elink_cb = function(lapl_id: hid_t; func: PH5L_elink_traverse_t; op_data: PPointer): herr_t; cdecl;
    TH5Pset_copy_object = function(plist_id: hid_t; crt_intmd: Cardinal): herr_t; cdecl;
    TH5Pget_copy_object = function(plist_id: hid_t; crt_intmd: PCardinal): herr_t; cdecl;
    TH5Padd_merge_committed_dtype_path = function(plist_id: hid_t; path: PAnsiChar): herr_t; cdecl;
    TH5Pfree_merge_committed_dtype_paths = function(plist_id: hid_t): herr_t; cdecl;
    TH5Pset_mcdt_search_cb = function(plist_id: hid_t; func: H5O_mcdt_search_cb_t; op_data: Pointer): herr_t; cdecl;
    TH5Pget_mcdt_search_cb = function(plist_id: hid_t; func: PH5O_mcdt_search_cb_t; op_data: PPointer): herr_t; cdecl;

  private
    FHandle: THandle;

    FH5open: TH5open;
    FH5close: TH5close;
    FH5dont_atexit: TH5dont_atexit;
    FH5garbage_collect: TH5garbage_collect;
    FH5set_free_list_limits: TH5set_free_list_limits;
    FH5get_libversion: TH5get_libversion;
    FH5check_version: TH5check_version;
    FH5is_library_threadsafe: TH5is_library_threadsafe;
    FH5free_memory: TH5free_memory;
    FH5allocate_memory: TH5allocate_memory;
    FH5resize_memory: TH5resize_memory;
    FH5Iregister: TH5Iregister;
    FH5Iobject_verify: TH5Iobject_verify;
    FH5Iremove_verify: TH5Iremove_verify;
    FH5Iget_type: TH5Iget_type;
    FH5Iget_file_id: TH5Iget_file_id;
    FH5Iget_name: TH5Iget_name;
    FH5Iinc_ref: TH5Iinc_ref;
    FH5Idec_ref: TH5Idec_ref;
    FH5Iget_ref: TH5Iget_ref;
    FH5Iregister_type: TH5Iregister_type;
    FH5Iclear_type: TH5Iclear_type;
    FH5Idestroy_type: TH5Idestroy_type;
    FH5Iinc_type_ref: TH5Iinc_type_ref;
    FH5Idec_type_ref: TH5Idec_type_ref;
    FH5Iget_type_ref: TH5Iget_type_ref;
    FH5Isearch: TH5Isearch;
    FH5Inmembers: TH5Inmembers;
    FH5Itype_exists: TH5Itype_exists;
    FH5Iis_valid: TH5Iis_valid;
    FH5Zregister: TH5Zregister;
    FH5Zunregister: TH5Zunregister;
    FH5Zfilter_avail: TH5Zfilter_avail;
    FH5Zget_filter_info: TH5Zget_filter_info;
    FH5PLset_loading_state: TH5PLset_loading_state;
    FH5PLget_loading_state: TH5PLget_loading_state;
    FH5T_IEEE_F32BE: hid_t;
    FH5T_IEEE_F32LE: hid_t;
    FH5T_IEEE_F64BE: hid_t;
    FH5T_IEEE_F64LE: hid_t;
    FH5T_STD_I8BE: hid_t;
    FH5T_STD_I8LE: hid_t;
    FH5T_STD_I16BE: hid_t;
    FH5T_STD_I16LE: hid_t;
    FH5T_STD_I32BE: hid_t;
    FH5T_STD_I32LE: hid_t;
    FH5T_STD_I64BE: hid_t;
    FH5T_STD_I64LE: hid_t;
    FH5T_STD_U8BE: hid_t;
    FH5T_STD_U8LE: hid_t;
    FH5T_STD_U16BE: hid_t;
    FH5T_STD_U16LE: hid_t;
    FH5T_STD_U32BE: hid_t;
    FH5T_STD_U32LE: hid_t;
    FH5T_STD_U64BE: hid_t;
    FH5T_STD_U64LE: hid_t;
    FH5T_STD_B8BE: hid_t;
    FH5T_STD_B8LE: hid_t;
    FH5T_STD_B16BE: hid_t;
    FH5T_STD_B16LE: hid_t;
    FH5T_STD_B32BE: hid_t;
    FH5T_STD_B32LE: hid_t;
    FH5T_STD_B64BE: hid_t;
    FH5T_STD_B64LE: hid_t;
    FH5T_STD_REF_OBJ: hid_t;
    FH5T_STD_REF_DSETREG: hid_t;
    FH5T_UNIX_D32BE: hid_t;
    FH5T_UNIX_D32LE: hid_t;
    FH5T_UNIX_D64BE: hid_t;
    FH5T_UNIX_D64LE: hid_t;
    FH5T_C_S1: hid_t;
    FH5T_FORTRAN_S1: hid_t;
    FH5T_VAX_F32: hid_t;
    FH5T_VAX_F64: hid_t;
    FH5T_NATIVE_SCHAR: hid_t;
    FH5T_NATIVE_UCHAR: hid_t;
    FH5T_NATIVE_SHORT: hid_t;
    FH5T_NATIVE_USHORT: hid_t;
    FH5T_NATIVE_INT: hid_t;
    FH5T_NATIVE_UINT: hid_t;
    FH5T_NATIVE_LONG: hid_t;
    FH5T_NATIVE_ULONG: hid_t;
    FH5T_NATIVE_LLONG: hid_t;
    FH5T_NATIVE_ULLONG: hid_t;
    FH5T_NATIVE_FLOAT: hid_t;
    FH5T_NATIVE_DOUBLE: hid_t;
    FH5T_NATIVE_B8: hid_t;
    FH5T_NATIVE_B16: hid_t;
    FH5T_NATIVE_B32: hid_t;
    FH5T_NATIVE_B64: hid_t;
    FH5T_NATIVE_OPAQUE: hid_t;
    FH5T_NATIVE_HADDR: hid_t;
    FH5T_NATIVE_HSIZE: hid_t;
    FH5T_NATIVE_HSSIZE: hid_t;
    FH5T_NATIVE_HERR: hid_t;
    FH5T_NATIVE_HBOOL: hid_t;
    FH5T_NATIVE_INT8: hid_t;
    FH5T_NATIVE_UINT8: hid_t;
    FH5T_NATIVE_INT_LEAST8: hid_t;
    FH5T_NATIVE_UINT_LEAST8: hid_t;
    FH5T_NATIVE_INT_FAST8: hid_t;
    FH5T_NATIVE_UINT_FAST8: hid_t;
    FH5T_NATIVE_INT16: hid_t;
    FH5T_NATIVE_UINT16: hid_t;
    FH5T_NATIVE_INT_LEAST16: hid_t;
    FH5T_NATIVE_UINT_LEAST16: hid_t;
    FH5T_NATIVE_INT_FAST16: hid_t;
    FH5T_NATIVE_UINT_FAST16: hid_t;
    FH5T_NATIVE_INT32: hid_t;
    FH5T_NATIVE_UINT32: hid_t;
    FH5T_NATIVE_INT_LEAST32: hid_t;
    FH5T_NATIVE_UINT_LEAST32: hid_t;
    FH5T_NATIVE_INT_FAST32: hid_t;
    FH5T_NATIVE_UINT_FAST32: hid_t;
    FH5T_NATIVE_INT64: hid_t;
    FH5T_NATIVE_UINT64: hid_t;
    FH5T_NATIVE_INT_LEAST64: hid_t;
    FH5T_NATIVE_UINT_LEAST64: hid_t;
    FH5T_NATIVE_INT_FAST64: hid_t;
    FH5T_NATIVE_UINT_FAST64: hid_t;
    FH5Tcreate: TH5Tcreate;
    FH5Tcopy: TH5Tcopy;
    FH5Tclose: TH5Tclose;
    FH5Tequal: TH5Tequal;
    FH5Tlock: TH5Tlock;
    FH5Tcommit2: TH5Tcommit2;
    FH5Topen2: TH5Topen2;
    FH5Tcommit_anon: TH5Tcommit_anon;
    FH5Tget_create_plist: TH5Tget_create_plist;
    FH5Tcommitted: TH5Tcommitted;
    FH5Tencode: TH5Tencode;
    FH5Tdecode: TH5Tdecode;
    FH5Tflush: TH5Tflush;
    FH5Trefresh: TH5Trefresh;
    FH5Tinsert: TH5Tinsert;
    FH5Tpack: TH5Tpack;
    FH5Tenum_create: TH5Tenum_create;
    FH5Tenum_insert: TH5Tenum_insert;
    FH5Tenum_nameof: TH5Tenum_nameof;
    FH5Tenum_valueof: TH5Tenum_valueof;
    FH5Tvlen_create: TH5Tvlen_create;
    FH5Tarray_create2: TH5Tarray_create2;
    FH5Tget_array_ndims: TH5Tget_array_ndims;
    FH5Tget_array_dims2: TH5Tget_array_dims2;
    FH5Tset_tag: TH5Tset_tag;
    FH5Tget_tag: TH5Tget_tag;
    FH5Tget_super: TH5Tget_super;
    FH5Tget_class: TH5Tget_class;
    FH5Tdetect_class: TH5Tdetect_class;
    FH5Tget_size: TH5Tget_size;
    FH5Tget_order: TH5Tget_order;
    FH5Tget_precision: TH5Tget_precision;
    FH5Tget_offset: TH5Tget_offset;
    FH5Tget_pad: TH5Tget_pad;
    FH5Tget_sign: TH5Tget_sign;
    FH5Tget_fields: TH5Tget_fields;
    FH5Tget_ebias: TH5Tget_ebias;
    FH5Tget_norm: TH5Tget_norm;
    FH5Tget_inpad: TH5Tget_inpad;
    FH5Tget_strpad: TH5Tget_strpad;
    FH5Tget_nmembers: TH5Tget_nmembers;
    FH5Tget_member_name: TH5Tget_member_name;
    FH5Tget_member_index: TH5Tget_member_index;
    FH5Tget_member_offset: TH5Tget_member_offset;
    FH5Tget_member_class: TH5Tget_member_class;
    FH5Tget_member_type: TH5Tget_member_type;
    FH5Tget_member_value: TH5Tget_member_value;
    FH5Tget_cset: TH5Tget_cset;
    FH5Tis_variable_str: TH5Tis_variable_str;
    FH5Tget_native_type: TH5Tget_native_type;
    FH5Tset_size: TH5Tset_size;
    FH5Tset_order: TH5Tset_order;
    FH5Tset_precision: TH5Tset_precision;
    FH5Tset_offset: TH5Tset_offset;
    FH5Tset_pad: TH5Tset_pad;
    FH5Tset_sign: TH5Tset_sign;
    FH5Tset_fields: TH5Tset_fields;
    FH5Tset_ebias: TH5Tset_ebias;
    FH5Tset_norm: TH5Tset_norm;
    FH5Tset_inpad: TH5Tset_inpad;
    FH5Tset_cset: TH5Tset_cset;
    FH5Tset_strpad: TH5Tset_strpad;
    FH5Tregister: TH5Tregister;
    FH5Tunregister: TH5Tunregister;
    FH5Tfind: TH5Tfind;
    FH5Tcompiler_conv: TH5Tcompiler_conv;
    FH5Tconvert: TH5Tconvert;
    FH5Dcreate2: TH5Dcreate2;
    FH5Dcreate_anon: TH5Dcreate_anon;
    FH5Dopen2: TH5Dopen2;
    FH5Dclose: TH5Dclose;
    FH5Dget_space: TH5Dget_space;
    FH5Dget_space_status: TH5Dget_space_status;
    FH5Dget_type: TH5Dget_type;
    FH5Dget_create_plist: TH5Dget_create_plist;
    FH5Dget_access_plist: TH5Dget_access_plist;
    FH5Dget_storage_size: TH5Dget_storage_size;
    FH5Dget_offset: TH5Dget_offset;
    FH5Dread: TH5Dread;
    FH5Dwrite: TH5Dwrite;
    FH5Diterate: TH5Diterate;
    FH5Dvlen_reclaim: TH5Dvlen_reclaim;
    FH5Dvlen_get_buf_size: TH5Dvlen_get_buf_size;
    FH5Dfill: TH5Dfill;
    FH5Dset_extent: TH5Dset_extent;
    FH5Dflush: TH5Dflush;
    FH5Drefresh: TH5Drefresh;
    FH5Dscatter: TH5Dscatter;
    FH5Dgather: TH5Dgather;
    FH5Ddebug: TH5Ddebug;
    FH5Dformat_convert: TH5Dformat_convert;
    FH5Dget_chunk_index_type: TH5Dget_chunk_index_type;
    FH5E_ERR_CLS: hid_t;
    FH5Eregister_class: TH5Eregister_class;
    FH5Eunregister_class: TH5Eunregister_class;
    FH5Eclose_msg: TH5Eclose_msg;
    FH5Ecreate_msg: TH5Ecreate_msg;
    FH5Ecreate_stack: TH5Ecreate_stack;
    FH5Eget_current_stack: TH5Eget_current_stack;
    FH5Eclose_stack: TH5Eclose_stack;
    FH5Eget_class_name: TH5Eget_class_name;
    FH5Eset_current_stack: TH5Eset_current_stack;
    // FH5Epush2: TH5Epush2;
    FH5Epop: TH5Epop;
    FH5Eprint2: TH5Eprint2;
    FH5Ewalk2: TH5Ewalk2;
    FH5Eget_auto2: TH5Eget_auto2;
    FH5Eset_auto2: TH5Eset_auto2;
    FH5Eclear2: TH5Eclear2;
    FH5Eauto_is_v2: TH5Eauto_is_v2;
    FH5Eget_msg: TH5Eget_msg;
    FH5Eget_num: TH5Eget_num;
    FH5Screate: TH5Screate;
    FH5Screate_simple: TH5Screate_simple;
    FH5Sset_extent_simple: TH5Sset_extent_simple;
    FH5Scopy: TH5Scopy;
    FH5Sclose: TH5Sclose;
    FH5Sencode: TH5Sencode;
    FH5Sdecode: TH5Sdecode;
    FH5Sget_simple_extent_npoints: TH5Sget_simple_extent_npoints;
    FH5Sget_simple_extent_ndims: TH5Sget_simple_extent_ndims;
    FH5Sget_simple_extent_dims: TH5Sget_simple_extent_dims;
    FH5Sis_simple: TH5Sis_simple;
    FH5Sget_select_npoints: TH5Sget_select_npoints;
    FH5Sselect_hyperslab: TH5Sselect_hyperslab;
    FH5Sselect_elements: TH5Sselect_elements;
    FH5Sget_simple_extent_type: TH5Sget_simple_extent_type;
    FH5Sset_extent_none: TH5Sset_extent_none;
    FH5Sextent_copy: TH5Sextent_copy;
    FH5Sextent_equal: TH5Sextent_equal;
    FH5Sselect_all: TH5Sselect_all;
    FH5Sselect_none: TH5Sselect_none;
    FH5Soffset_simple: TH5Soffset_simple;
    FH5Sselect_valid: TH5Sselect_valid;
    FH5Sis_regular_hyperslab: TH5Sis_regular_hyperslab;
    FH5Sget_regular_hyperslab: TH5Sget_regular_hyperslab;
    FH5Sget_select_hyper_nblocks: TH5Sget_select_hyper_nblocks;
    FH5Sget_select_elem_npoints: TH5Sget_select_elem_npoints;
    FH5Sget_select_hyper_blocklist: TH5Sget_select_hyper_blocklist;
    FH5Sget_select_elem_pointlist: TH5Sget_select_elem_pointlist;
    FH5Sget_select_bounds: TH5Sget_select_bounds;
    FH5Sget_select_type: TH5Sget_select_type;
    FH5Lmove: TH5Lmove;
    FH5Lcopy: TH5Lcopy;
    FH5Lcreate_hard: TH5Lcreate_hard;
    FH5Lcreate_soft: TH5Lcreate_soft;
    FH5Ldelete: TH5Ldelete;
    FH5Ldelete_by_idx: TH5Ldelete_by_idx;
    FH5Lget_val: TH5Lget_val;
    FH5Lget_val_by_idx: TH5Lget_val_by_idx;
    FH5Lexists: TH5Lexists;
    FH5Lget_info: TH5Lget_info;
    FH5Lget_info_by_idx: TH5Lget_info_by_idx;
    FH5Lget_name_by_idx: TH5Lget_name_by_idx;
    FH5Literate: TH5Literate;
    FH5Literate_by_name: TH5Literate_by_name;
    FH5Lvisit: TH5Lvisit;
    FH5Lvisit_by_name: TH5Lvisit_by_name;
    FH5Lcreate_ud: TH5Lcreate_ud;
    FH5Lregister: TH5Lregister;
    FH5Lunregister: TH5Lunregister;
    FH5Lis_registered: TH5Lis_registered;
    FH5Lunpack_elink_val: TH5Lunpack_elink_val;
    FH5Lcreate_external: TH5Lcreate_external;
    FH5Oopen: TH5Oopen;
    FH5Oopen_by_addr: TH5Oopen_by_addr;
    FH5Oopen_by_idx: TH5Oopen_by_idx;
    FH5Oexists_by_name: TH5Oexists_by_name;
    FH5Oget_info: TH5Oget_info;
    FH5Oget_info_by_name: TH5Oget_info_by_name;
    FH5Oget_info_by_idx: TH5Oget_info_by_idx;
    FH5Olink: TH5Olink;
    FH5Oincr_refcount: TH5Oincr_refcount;
    FH5Odecr_refcount: TH5Odecr_refcount;
    FH5Ocopy: TH5Ocopy;
    FH5Oset_comment: TH5Oset_comment;
    FH5Oset_comment_by_name: TH5Oset_comment_by_name;
    FH5Oget_comment: TH5Oget_comment;
    FH5Oget_comment_by_name: TH5Oget_comment_by_name;
    FH5Ovisit: TH5Ovisit;
    FH5Ovisit_by_name: TH5Ovisit_by_name;
    FH5Oclose: TH5Oclose;
    FH5Oflush: TH5Oflush;
    FH5Orefresh: TH5Orefresh;
    FH5Odisable_mdc_flushes: TH5Odisable_mdc_flushes;
    FH5Oenable_mdc_flushes: TH5Oenable_mdc_flushes;
    FH5Oare_mdc_flushes_disabled: TH5Oare_mdc_flushes_disabled;
    FH5Fis_hdf5: TH5Fis_hdf5;
    FH5Fcreate: TH5Fcreate;
    FH5Fopen: TH5Fopen;
    FH5Freopen: TH5Freopen;
    FH5Fflush: TH5Fflush;
    FH5Fclose: TH5Fclose;
    FH5Fget_create_plist: TH5Fget_create_plist;
    FH5Fget_access_plist: TH5Fget_access_plist;
    FH5Fget_intent: TH5Fget_intent;
    FH5Fget_obj_count: TH5Fget_obj_count;
    FH5Fget_obj_ids: TH5Fget_obj_ids;
    FH5Fget_vfd_handle: TH5Fget_vfd_handle;
    FH5Fmount: TH5Fmount;
    FH5Funmount: TH5Funmount;
    FH5Fget_freespace: TH5Fget_freespace;
    FH5Fget_filesize: TH5Fget_filesize;
    FH5Fget_file_image: TH5Fget_file_image;
    FH5Fget_mdc_config: TH5Fget_mdc_config;
    FH5Fset_mdc_config: TH5Fset_mdc_config;
    FH5Fget_mdc_hit_rate: TH5Fget_mdc_hit_rate;
    FH5Fget_mdc_size: TH5Fget_mdc_size;
    FH5Freset_mdc_hit_rate_stats: TH5Freset_mdc_hit_rate_stats;
    FH5Fget_name: TH5Fget_name;
    FH5Fget_info2: TH5Fget_info2;
    FH5Fget_metadata_read_retry_info: TH5Fget_metadata_read_retry_info;
    FH5Fstart_swmr_write: TH5Fstart_swmr_write;
    FH5Fget_free_sections: TH5Fget_free_sections;
    FH5Fclear_elink_file_cache: TH5Fclear_elink_file_cache;
    FH5Fstart_mdc_logging: TH5Fstart_mdc_logging;
    FH5Fstop_mdc_logging: TH5Fstop_mdc_logging;
    FH5Fget_mdc_logging_status: TH5Fget_mdc_logging_status;
    FH5Fformat_convert: TH5Fformat_convert;
    FH5Acreate2: TH5Acreate2;
    FH5Acreate_by_name: TH5Acreate_by_name;
    FH5Aopen: TH5Aopen;
    FH5Aopen_by_name: TH5Aopen_by_name;
    FH5Aopen_by_idx: TH5Aopen_by_idx;
    FH5Awrite: TH5Awrite;
    FH5Aread: TH5Aread;
    FH5Aclose: TH5Aclose;
    FH5Aget_space: TH5Aget_space;
    FH5Aget_type: TH5Aget_type;
    FH5Aget_create_plist: TH5Aget_create_plist;
    FH5Aget_name: TH5Aget_name;
    FH5Aget_name_by_idx: TH5Aget_name_by_idx;
    FH5Aget_storage_size: TH5Aget_storage_size;
    FH5Aget_info: TH5Aget_info;
    FH5Aget_info_by_name: TH5Aget_info_by_name;
    FH5Aget_info_by_idx: TH5Aget_info_by_idx;
    FH5Arename: TH5Arename;
    FH5Arename_by_name: TH5Arename_by_name;
    FH5Aiterate2: TH5Aiterate2;
    FH5Aiterate_by_name: TH5Aiterate_by_name;
    FH5Adelete: TH5Adelete;
    FH5Adelete_by_name: TH5Adelete_by_name;
    FH5Adelete_by_idx: TH5Adelete_by_idx;
    FH5Aexists: TH5Aexists;
    FH5Aexists_by_name: TH5Aexists_by_name;
    FH5FDregister: TH5FDregister;
    FH5FDunregister: TH5FDunregister;
    FH5FDopen: TH5FDopen;
    FH5FDclose: TH5FDclose;
    FH5FDcmp: TH5FDcmp;
    FH5FDquery: TH5FDquery;
    FH5FDalloc: TH5FDalloc;
    FH5FDfree: TH5FDfree;
    FH5FDget_eoa: TH5FDget_eoa;
    FH5FDset_eoa: TH5FDset_eoa;
    FH5FDget_eof: TH5FDget_eof;
    FH5FDget_vfd_handle: TH5FDget_vfd_handle;
    FH5FDread: TH5FDread;
    FH5FDwrite: TH5FDwrite;
    FH5FDflush: TH5FDflush;
    FH5FDtruncate: TH5FDtruncate;
    FH5FDlock: TH5FDlock;
    FH5FDunlock: TH5FDunlock;
    FH5Gcreate2: TH5Gcreate2;
    FH5Gcreate_anon: TH5Gcreate_anon;
    FH5Gopen2: TH5Gopen2;
    FH5Gget_create_plist: TH5Gget_create_plist;
    FH5Gget_info: TH5Gget_info;
    FH5Gget_info_by_name: TH5Gget_info_by_name;
    FH5Gget_info_by_idx: TH5Gget_info_by_idx;
    FH5Gclose: TH5Gclose;
    FH5Gflush: TH5Gflush;
    FH5Grefresh: TH5Grefresh;
    FH5Rcreate: TH5Rcreate;
    FH5Rdereference2: TH5Rdereference2;
    FH5Rget_region: TH5Rget_region;
    FH5Rget_obj_type2: TH5Rget_obj_type2;
    FH5Rget_name: TH5Rget_name;
    FH5P_CLS_ROOT_ID: hid_t;
    FH5P_CLS_OBJECT_CREATE_ID: hid_t;
    FH5P_CLS_FILE_CREATE_ID: hid_t;
    FH5P_CLS_FILE_ACCESS_ID: hid_t;
    FH5P_CLS_DATASET_CREATE_ID: hid_t;
    FH5P_CLS_DATASET_ACCESS_ID: hid_t;
    FH5P_CLS_DATASET_XFER_ID: hid_t;
    FH5P_CLS_FILE_MOUNT_ID: hid_t;
    FH5P_CLS_GROUP_CREATE_ID: hid_t;
    FH5P_CLS_GROUP_ACCESS_ID: hid_t;
    FH5P_CLS_DATATYPE_CREATE_ID: hid_t;
    FH5P_CLS_DATATYPE_ACCESS_ID: hid_t;
    FH5P_CLS_STRING_CREATE_ID: hid_t;
    FH5P_CLS_ATTRIBUTE_CREATE_ID: hid_t;
    FH5P_CLS_ATTRIBUTE_ACCESS_ID: hid_t;
    FH5P_CLS_OBJECT_COPY_ID: hid_t;
    FH5P_CLS_LINK_CREATE_ID: hid_t;
    FH5P_CLS_LINK_ACCESS_ID: hid_t;
    FH5P_LST_FILE_CREATE_ID: hid_t;
    FH5P_LST_FILE_ACCESS_ID: hid_t;
    FH5P_LST_DATASET_CREATE_ID: hid_t;
    FH5P_LST_DATASET_ACCESS_ID: hid_t;
    FH5P_LST_DATASET_XFER_ID: hid_t;
    FH5P_LST_FILE_MOUNT_ID: hid_t;
    FH5P_LST_GROUP_CREATE_ID: hid_t;
    FH5P_LST_GROUP_ACCESS_ID: hid_t;
    FH5P_LST_DATATYPE_CREATE_ID: hid_t;
    FH5P_LST_DATATYPE_ACCESS_ID: hid_t;
    FH5P_LST_ATTRIBUTE_CREATE_ID: hid_t;
    FH5P_LST_ATTRIBUTE_ACCESS_ID: hid_t;
    FH5P_LST_OBJECT_COPY_ID: hid_t;
    FH5P_LST_LINK_CREATE_ID: hid_t;
    FH5P_LST_LINK_ACCESS_ID: hid_t;
    FH5Pcreate_class: TH5Pcreate_class;
    FH5Pget_class_name: TH5Pget_class_name;
    FH5Pcreate: TH5Pcreate;
    FH5Pregister2: TH5Pregister2;
    FH5Pinsert2: TH5Pinsert2;
    FH5Pset: TH5Pset;
    FH5Pexist: TH5Pexist;
    FH5Pencode: TH5Pencode;
    FH5Pdecode: TH5Pdecode;
    FH5Pget_size: TH5Pget_size;
    FH5Pget_nprops: TH5Pget_nprops;
    FH5Pget_class: TH5Pget_class;
    FH5Pget_class_parent: TH5Pget_class_parent;
    FH5Pget: TH5Pget;
    FH5Pequal: TH5Pequal;
    FH5Pisa_class: TH5Pisa_class;
    FH5Piterate: TH5Piterate;
    FH5Pcopy_prop: TH5Pcopy_prop;
    FH5Premove: TH5Premove;
    FH5Punregister: TH5Punregister;
    FH5Pclose_class: TH5Pclose_class;
    FH5Pclose: TH5Pclose;
    FH5Pcopy: TH5Pcopy;
    FH5Pset_attr_phase_change: TH5Pset_attr_phase_change;
    FH5Pget_attr_phase_change: TH5Pget_attr_phase_change;
    FH5Pset_attr_creation_order: TH5Pset_attr_creation_order;
    FH5Pget_attr_creation_order: TH5Pget_attr_creation_order;
    FH5Pset_obj_track_times: TH5Pset_obj_track_times;
    FH5Pget_obj_track_times: TH5Pget_obj_track_times;
    FH5Pmodify_filter: TH5Pmodify_filter;
    FH5Pset_filter: TH5Pset_filter;
    FH5Pget_nfilters: TH5Pget_nfilters;
    FH5Pget_filter2: TH5Pget_filter2;
    FH5Pget_filter_by_id2: TH5Pget_filter_by_id2;
    FH5Pall_filters_avail: TH5Pall_filters_avail;
    FH5Premove_filter: TH5Premove_filter;
    FH5Pset_deflate: TH5Pset_deflate;
    FH5Pset_fletcher32: TH5Pset_fletcher32;
    FH5Pset_userblock: TH5Pset_userblock;
    FH5Pget_userblock: TH5Pget_userblock;
    FH5Pset_sizes: TH5Pset_sizes;
    FH5Pget_sizes: TH5Pget_sizes;
    FH5Pset_sym_k: TH5Pset_sym_k;
    FH5Pget_sym_k: TH5Pget_sym_k;
    FH5Pset_istore_k: TH5Pset_istore_k;
    FH5Pget_istore_k: TH5Pget_istore_k;
    FH5Pset_shared_mesg_nindexes: TH5Pset_shared_mesg_nindexes;
    FH5Pget_shared_mesg_nindexes: TH5Pget_shared_mesg_nindexes;
    FH5Pset_shared_mesg_index: TH5Pset_shared_mesg_index;
    FH5Pget_shared_mesg_index: TH5Pget_shared_mesg_index;
    FH5Pset_shared_mesg_phase_change: TH5Pset_shared_mesg_phase_change;
    FH5Pget_shared_mesg_phase_change: TH5Pget_shared_mesg_phase_change;
    FH5Pset_file_space: TH5Pset_file_space;
    FH5Pget_file_space: TH5Pget_file_space;
    FH5Pset_alignment: TH5Pset_alignment;
    FH5Pget_alignment: TH5Pget_alignment;
    FH5Pset_driver: TH5Pset_driver;
    FH5Pget_driver: TH5Pget_driver;
    FH5Pget_driver_info: TH5Pget_driver_info;
    FH5Pset_family_offset: TH5Pset_family_offset;
    FH5Pget_family_offset: TH5Pget_family_offset;
    FH5Pset_multi_type: TH5Pset_multi_type;
    FH5Pget_multi_type: TH5Pget_multi_type;
    FH5Pset_cache: TH5Pset_cache;
    FH5Pget_cache: TH5Pget_cache;
    FH5Pset_mdc_config: TH5Pset_mdc_config;
    FH5Pget_mdc_config: TH5Pget_mdc_config;
    FH5Pset_gc_references: TH5Pset_gc_references;
    FH5Pget_gc_references: TH5Pget_gc_references;
    FH5Pset_fclose_degree: TH5Pset_fclose_degree;
    FH5Pget_fclose_degree: TH5Pget_fclose_degree;
    FH5Pset_meta_block_size: TH5Pset_meta_block_size;
    FH5Pget_meta_block_size: TH5Pget_meta_block_size;
    FH5Pset_sieve_buf_size: TH5Pset_sieve_buf_size;
    FH5Pget_sieve_buf_size: TH5Pget_sieve_buf_size;
    FH5Pset_small_data_block_size: TH5Pset_small_data_block_size;
    FH5Pget_small_data_block_size: TH5Pget_small_data_block_size;
    FH5Pset_libver_bounds: TH5Pset_libver_bounds;
    FH5Pget_libver_bounds: TH5Pget_libver_bounds;
    FH5Pset_elink_file_cache_size: TH5Pset_elink_file_cache_size;
    FH5Pget_elink_file_cache_size: TH5Pget_elink_file_cache_size;
    FH5Pset_file_image: TH5Pset_file_image;
    FH5Pget_file_image: TH5Pget_file_image;
    FH5Pset_file_image_callbacks: TH5Pset_file_image_callbacks;
    FH5Pget_file_image_callbacks: TH5Pget_file_image_callbacks;
    FH5Pset_core_write_tracking: TH5Pset_core_write_tracking;
    FH5Pget_core_write_tracking: TH5Pget_core_write_tracking;
    FH5Pset_metadata_read_attempts: TH5Pset_metadata_read_attempts;
    FH5Pget_metadata_read_attempts: TH5Pget_metadata_read_attempts;
    FH5Pset_object_flush_cb: TH5Pset_object_flush_cb;
    FH5Pget_object_flush_cb: TH5Pget_object_flush_cb;
    FH5Pset_mdc_log_options: TH5Pset_mdc_log_options;
    FH5Pget_mdc_log_options: TH5Pget_mdc_log_options;
    FH5Pset_layout: TH5Pset_layout;
    FH5Pget_layout: TH5Pget_layout;
    FH5Pset_chunk: TH5Pset_chunk;
    FH5Pget_chunk: TH5Pget_chunk;
    FH5Pset_virtual: TH5Pset_virtual;
    FH5Pget_virtual_count: TH5Pget_virtual_count;
    FH5Pget_virtual_vspace: TH5Pget_virtual_vspace;
    FH5Pget_virtual_srcspace: TH5Pget_virtual_srcspace;
    FH5Pget_virtual_filename: TH5Pget_virtual_filename;
    FH5Pget_virtual_dsetname: TH5Pget_virtual_dsetname;
    FH5Pset_external: TH5Pset_external;
    FH5Pset_chunk_opts: TH5Pset_chunk_opts;
    FH5Pget_chunk_opts: TH5Pget_chunk_opts;
    FH5Pget_external_count: TH5Pget_external_count;
    FH5Pget_external: TH5Pget_external;
    FH5Pset_szip: TH5Pset_szip;
    FH5Pset_shuffle: TH5Pset_shuffle;
    FH5Pset_nbit: TH5Pset_nbit;
    FH5Pset_scaleoffset: TH5Pset_scaleoffset;
    FH5Pset_fill_value: TH5Pset_fill_value;
    FH5Pget_fill_value: TH5Pget_fill_value;
    FH5Pfill_value_defined: TH5Pfill_value_defined;
    FH5Pset_alloc_time: TH5Pset_alloc_time;
    FH5Pget_alloc_time: TH5Pget_alloc_time;
    FH5Pset_fill_time: TH5Pset_fill_time;
    FH5Pget_fill_time: TH5Pget_fill_time;
    FH5Pset_chunk_cache: TH5Pset_chunk_cache;
    FH5Pget_chunk_cache: TH5Pget_chunk_cache;
    FH5Pset_virtual_view: TH5Pset_virtual_view;
    FH5Pget_virtual_view: TH5Pget_virtual_view;
    FH5Pset_virtual_printf_gap: TH5Pset_virtual_printf_gap;
    FH5Pget_virtual_printf_gap: TH5Pget_virtual_printf_gap;
    FH5Pset_append_flush: TH5Pset_append_flush;
    FH5Pget_append_flush: TH5Pget_append_flush;
    FH5Pset_efile_prefix: TH5Pset_efile_prefix;
    FH5Pget_efile_prefix: TH5Pget_efile_prefix;
    FH5Pset_data_transform: TH5Pset_data_transform;
    FH5Pget_data_transform: TH5Pget_data_transform;
    FH5Pset_buffer: TH5Pset_buffer;
    FH5Pget_buffer: TH5Pget_buffer;
    FH5Pset_preserve: TH5Pset_preserve;
    FH5Pget_preserve: TH5Pget_preserve;
    FH5Pset_edc_check: TH5Pset_edc_check;
    FH5Pget_edc_check: TH5Pget_edc_check;
    FH5Pset_filter_callback: TH5Pset_filter_callback;
    FH5Pset_btree_ratios: TH5Pset_btree_ratios;
    FH5Pget_btree_ratios: TH5Pget_btree_ratios;
    FH5Pset_vlen_mem_manager: TH5Pset_vlen_mem_manager;
    FH5Pget_vlen_mem_manager: TH5Pget_vlen_mem_manager;
    FH5Pset_hyper_vector_size: TH5Pset_hyper_vector_size;
    FH5Pget_hyper_vector_size: TH5Pget_hyper_vector_size;
    FH5Pset_type_conv_cb: TH5Pset_type_conv_cb;
    FH5Pget_type_conv_cb: TH5Pget_type_conv_cb;
    FH5Pset_create_intermediate_group: TH5Pset_create_intermediate_group;
    FH5Pget_create_intermediate_group: TH5Pget_create_intermediate_group;
    FH5Pset_local_heap_size_hint: TH5Pset_local_heap_size_hint;
    FH5Pget_local_heap_size_hint: TH5Pget_local_heap_size_hint;
    FH5Pset_link_phase_change: TH5Pset_link_phase_change;
    FH5Pget_link_phase_change: TH5Pget_link_phase_change;
    FH5Pset_est_link_info: TH5Pset_est_link_info;
    FH5Pget_est_link_info: TH5Pget_est_link_info;
    FH5Pset_link_creation_order: TH5Pset_link_creation_order;
    FH5Pget_link_creation_order: TH5Pget_link_creation_order;
    FH5Pset_char_encoding: TH5Pset_char_encoding;
    FH5Pget_char_encoding: TH5Pget_char_encoding;
    FH5Pset_nlinks: TH5Pset_nlinks;
    FH5Pget_nlinks: TH5Pget_nlinks;
    FH5Pset_elink_prefix: TH5Pset_elink_prefix;
    FH5Pget_elink_prefix: TH5Pget_elink_prefix;
    FH5Pget_elink_fapl: TH5Pget_elink_fapl;
    FH5Pset_elink_fapl: TH5Pset_elink_fapl;
    FH5Pset_elink_acc_flags: TH5Pset_elink_acc_flags;
    FH5Pget_elink_acc_flags: TH5Pget_elink_acc_flags;
    FH5Pset_elink_cb: TH5Pset_elink_cb;
    FH5Pget_elink_cb: TH5Pget_elink_cb;
    FH5Pset_copy_object: TH5Pset_copy_object;
    FH5Pget_copy_object: TH5Pget_copy_object;
    FH5Padd_merge_committed_dtype_path: TH5Padd_merge_committed_dtype_path;
    FH5Pfree_merge_committed_dtype_paths: TH5Pfree_merge_committed_dtype_paths;
    FH5Pset_mcdt_search_cb: TH5Pset_mcdt_search_cb;
    FH5Pget_mcdt_search_cb: TH5Pget_mcdt_search_cb;

  public
    constructor Create(APath: string);
    destructor Destroy; override;

    property H5open: TH5open read FH5open;
    property H5close: TH5close read FH5close;
    property H5dont_atexit: TH5dont_atexit read FH5dont_atexit;
    property H5garbage_collect: TH5garbage_collect read FH5garbage_collect;
    property H5set_free_list_limits: TH5set_free_list_limits read FH5set_free_list_limits;
    property H5get_libversion: TH5get_libversion read FH5get_libversion;
    property H5check_version: TH5check_version read FH5check_version;
    property H5is_library_threadsafe: TH5is_library_threadsafe read FH5is_library_threadsafe;
    property H5free_memory: TH5free_memory read FH5free_memory;
    property H5allocate_memory: TH5allocate_memory read FH5allocate_memory;
    property H5resize_memory: TH5resize_memory read FH5resize_memory;
    property H5Iregister: TH5Iregister read FH5Iregister;
    property H5Iobject_verify: TH5Iobject_verify read FH5Iobject_verify;
    property H5Iremove_verify: TH5Iremove_verify read FH5Iremove_verify;
    property H5Iget_type: TH5Iget_type read FH5Iget_type;
    property H5Iget_file_id: TH5Iget_file_id read FH5Iget_file_id;
    property H5Iget_name: TH5Iget_name read FH5Iget_name;
    property H5Iinc_ref: TH5Iinc_ref read FH5Iinc_ref;
    property H5Idec_ref: TH5Idec_ref read FH5Idec_ref;
    property H5Iget_ref: TH5Iget_ref read FH5Iget_ref;
    property H5Iregister_type: TH5Iregister_type read FH5Iregister_type;
    property H5Iclear_type: TH5Iclear_type read FH5Iclear_type;
    property H5Idestroy_type: TH5Idestroy_type read FH5Idestroy_type;
    property H5Iinc_type_ref: TH5Iinc_type_ref read FH5Iinc_type_ref;
    property H5Idec_type_ref: TH5Idec_type_ref read FH5Idec_type_ref;
    property H5Iget_type_ref: TH5Iget_type_ref read FH5Iget_type_ref;
    property H5Isearch: TH5Isearch read FH5Isearch;
    property H5Inmembers: TH5Inmembers read FH5Inmembers;
    property H5Itype_exists: TH5Itype_exists read FH5Itype_exists;
    property H5Iis_valid: TH5Iis_valid read FH5Iis_valid;
    property H5Zregister: TH5Zregister read FH5Zregister;
    property H5Zunregister: TH5Zunregister read FH5Zunregister;
    property H5Zfilter_avail: TH5Zfilter_avail read FH5Zfilter_avail;
    property H5Zget_filter_info: TH5Zget_filter_info read FH5Zget_filter_info;
    property H5PLset_loading_state: TH5PLset_loading_state read FH5PLset_loading_state;
    property H5PLget_loading_state: TH5PLget_loading_state read FH5PLget_loading_state;
    property H5T_IEEE_F32BE: hid_t read FH5T_IEEE_F32BE;
    property H5T_IEEE_F32LE: hid_t read FH5T_IEEE_F32LE;
    property H5T_IEEE_F64BE: hid_t read FH5T_IEEE_F64BE;
    property H5T_IEEE_F64LE: hid_t read FH5T_IEEE_F64LE;
    property H5T_STD_I8BE: hid_t read FH5T_STD_I8BE;
    property H5T_STD_I8LE: hid_t read FH5T_STD_I8LE;
    property H5T_STD_I16BE: hid_t read FH5T_STD_I16BE;
    property H5T_STD_I16LE: hid_t read FH5T_STD_I16LE;
    property H5T_STD_I32BE: hid_t read FH5T_STD_I32BE;
    property H5T_STD_I32LE: hid_t read FH5T_STD_I32LE;
    property H5T_STD_I64BE: hid_t read FH5T_STD_I64BE;
    property H5T_STD_I64LE: hid_t read FH5T_STD_I64LE;
    property H5T_STD_U8BE: hid_t read FH5T_STD_U8BE;
    property H5T_STD_U8LE: hid_t read FH5T_STD_U8LE;
    property H5T_STD_U16BE: hid_t read FH5T_STD_U16BE;
    property H5T_STD_U16LE: hid_t read FH5T_STD_U16LE;
    property H5T_STD_U32BE: hid_t read FH5T_STD_U32BE;
    property H5T_STD_U32LE: hid_t read FH5T_STD_U32LE;
    property H5T_STD_U64BE: hid_t read FH5T_STD_U64BE;
    property H5T_STD_U64LE: hid_t read FH5T_STD_U64LE;
    property H5T_STD_B8BE: hid_t read FH5T_STD_B8BE;
    property H5T_STD_B8LE: hid_t read FH5T_STD_B8LE;
    property H5T_STD_B16BE: hid_t read FH5T_STD_B16BE;
    property H5T_STD_B16LE: hid_t read FH5T_STD_B16LE;
    property H5T_STD_B32BE: hid_t read FH5T_STD_B32BE;
    property H5T_STD_B32LE: hid_t read FH5T_STD_B32LE;
    property H5T_STD_B64BE: hid_t read FH5T_STD_B64BE;
    property H5T_STD_B64LE: hid_t read FH5T_STD_B64LE;
    property H5T_STD_REF_OBJ: hid_t read FH5T_STD_REF_OBJ;
    property H5T_STD_REF_DSETREG: hid_t read FH5T_STD_REF_DSETREG;
    property H5T_UNIX_D32BE: hid_t read FH5T_UNIX_D32BE;
    property H5T_UNIX_D32LE: hid_t read FH5T_UNIX_D32LE;
    property H5T_UNIX_D64BE: hid_t read FH5T_UNIX_D64BE;
    property H5T_UNIX_D64LE: hid_t read FH5T_UNIX_D64LE;
    property H5T_C_S1: hid_t read FH5T_C_S1;
    property H5T_FORTRAN_S1: hid_t read FH5T_FORTRAN_S1;
    property H5T_INTEL_I8: hid_t read FH5T_STD_I8LE;
    property H5T_INTEL_I16: hid_t read FH5T_STD_I16LE;
    property H5T_INTEL_I32: hid_t read FH5T_STD_I32LE;
    property H5T_INTEL_I64: hid_t read FH5T_STD_I64LE;
    property H5T_INTEL_U8: hid_t read FH5T_STD_U8LE;
    property H5T_INTEL_U16: hid_t read FH5T_STD_U16LE;
    property H5T_INTEL_U32: hid_t read FH5T_STD_U32LE;
    property H5T_INTEL_U64: hid_t read FH5T_STD_U64LE;
    property H5T_INTEL_B8: hid_t read FH5T_STD_B8LE;
    property H5T_INTEL_B16: hid_t read FH5T_STD_B16LE;
    property H5T_INTEL_B32: hid_t read FH5T_STD_B32LE;
    property H5T_INTEL_B64: hid_t read FH5T_STD_B64LE;
    property H5T_INTEL_F32: hid_t read FH5T_IEEE_F32LE;
    property H5T_INTEL_F64: hid_t read FH5T_IEEE_F64LE;
    property H5T_ALPHA_I8: hid_t read FH5T_STD_I8LE;
    property H5T_ALPHA_I16: hid_t read FH5T_STD_I16LE;
    property H5T_ALPHA_I32: hid_t read FH5T_STD_I32LE;
    property H5T_ALPHA_I64: hid_t read FH5T_STD_I64LE;
    property H5T_ALPHA_U8: hid_t read FH5T_STD_U8LE;
    property H5T_ALPHA_U16: hid_t read FH5T_STD_U16LE;
    property H5T_ALPHA_U32: hid_t read FH5T_STD_U32LE;
    property H5T_ALPHA_U64: hid_t read FH5T_STD_U64LE;
    property H5T_ALPHA_B8: hid_t read FH5T_STD_B8LE;
    property H5T_ALPHA_B16: hid_t read FH5T_STD_B16LE;
    property H5T_ALPHA_B32: hid_t read FH5T_STD_B32LE;
    property H5T_ALPHA_B64: hid_t read FH5T_STD_B64LE;
    property H5T_ALPHA_F32: hid_t read FH5T_IEEE_F32LE;
    property H5T_ALPHA_F64: hid_t read FH5T_IEEE_F64LE;
    property H5T_MIPS_I8: hid_t read FH5T_STD_I8BE;
    property H5T_MIPS_I16: hid_t read FH5T_STD_I16BE;
    property H5T_MIPS_I32: hid_t read FH5T_STD_I32BE;
    property H5T_MIPS_I64: hid_t read FH5T_STD_I64BE;
    property H5T_MIPS_U8: hid_t read FH5T_STD_U8BE;
    property H5T_MIPS_U16: hid_t read FH5T_STD_U16BE;
    property H5T_MIPS_U32: hid_t read FH5T_STD_U32BE;
    property H5T_MIPS_U64: hid_t read FH5T_STD_U64BE;
    property H5T_MIPS_B8: hid_t read FH5T_STD_B8BE;
    property H5T_MIPS_B16: hid_t read FH5T_STD_B16BE;
    property H5T_MIPS_B32: hid_t read FH5T_STD_B32BE;
    property H5T_MIPS_B64: hid_t read FH5T_STD_B64BE;
    property H5T_MIPS_F32: hid_t read FH5T_IEEE_F32BE;
    property H5T_MIPS_F64: hid_t read FH5T_IEEE_F64BE;
    property H5T_VAX_F32: hid_t read FH5T_VAX_F32;
    property H5T_VAX_F64: hid_t read FH5T_VAX_F64;
    property H5T_NATIVE_SCHAR: hid_t read FH5T_NATIVE_SCHAR;
    property H5T_NATIVE_UCHAR: hid_t read FH5T_NATIVE_UCHAR;
    property H5T_NATIVE_SHORT: hid_t read FH5T_NATIVE_SHORT;
    property H5T_NATIVE_USHORT: hid_t read FH5T_NATIVE_USHORT;
    property H5T_NATIVE_INT: hid_t read FH5T_NATIVE_INT;
    property H5T_NATIVE_UINT: hid_t read FH5T_NATIVE_UINT;
    property H5T_NATIVE_LONG: hid_t read FH5T_NATIVE_LONG;
    property H5T_NATIVE_ULONG: hid_t read FH5T_NATIVE_ULONG;
    property H5T_NATIVE_LLONG: hid_t read FH5T_NATIVE_LLONG;
    property H5T_NATIVE_ULLONG: hid_t read FH5T_NATIVE_ULLONG;
    property H5T_NATIVE_FLOAT: hid_t read FH5T_NATIVE_FLOAT;
    property H5T_NATIVE_DOUBLE: hid_t read FH5T_NATIVE_DOUBLE;
    property H5T_NATIVE_B8: hid_t read FH5T_NATIVE_B8;
    property H5T_NATIVE_B16: hid_t read FH5T_NATIVE_B16;
    property H5T_NATIVE_B32: hid_t read FH5T_NATIVE_B32;
    property H5T_NATIVE_B64: hid_t read FH5T_NATIVE_B64;
    property H5T_NATIVE_OPAQUE: hid_t read FH5T_NATIVE_OPAQUE;
    property H5T_NATIVE_HADDR: hid_t read FH5T_NATIVE_HADDR;
    property H5T_NATIVE_HSIZE: hid_t read FH5T_NATIVE_HSIZE;
    property H5T_NATIVE_HSSIZE: hid_t read FH5T_NATIVE_HSSIZE;
    property H5T_NATIVE_HERR: hid_t read FH5T_NATIVE_HERR;
    property H5T_NATIVE_HBOOL: hid_t read FH5T_NATIVE_HBOOL;
    property H5T_NATIVE_INT8: hid_t read FH5T_NATIVE_INT8;
    property H5T_NATIVE_UINT8: hid_t read FH5T_NATIVE_UINT8;
    property H5T_NATIVE_INT_LEAST8: hid_t read FH5T_NATIVE_INT_LEAST8;
    property H5T_NATIVE_UINT_LEAST8: hid_t read FH5T_NATIVE_UINT_LEAST8;
    property H5T_NATIVE_INT_FAST8: hid_t read FH5T_NATIVE_INT_FAST8;
    property H5T_NATIVE_UINT_FAST8: hid_t read FH5T_NATIVE_UINT_FAST8;
    property H5T_NATIVE_INT16: hid_t read FH5T_NATIVE_INT16;
    property H5T_NATIVE_UINT16: hid_t read FH5T_NATIVE_UINT16;
    property H5T_NATIVE_INT_LEAST16: hid_t read FH5T_NATIVE_INT_LEAST16;
    property H5T_NATIVE_UINT_LEAST16: hid_t read FH5T_NATIVE_UINT_LEAST16;
    property H5T_NATIVE_INT_FAST16: hid_t read FH5T_NATIVE_INT_FAST16;
    property H5T_NATIVE_UINT_FAST16: hid_t read FH5T_NATIVE_UINT_FAST16;
    property H5T_NATIVE_INT32: hid_t read FH5T_NATIVE_INT32;
    property H5T_NATIVE_UINT32: hid_t read FH5T_NATIVE_UINT32;
    property H5T_NATIVE_INT_LEAST32: hid_t read FH5T_NATIVE_INT_LEAST32;
    property H5T_NATIVE_UINT_LEAST32: hid_t read FH5T_NATIVE_UINT_LEAST32;
    property H5T_NATIVE_INT_FAST32: hid_t read FH5T_NATIVE_INT_FAST32;
    property H5T_NATIVE_UINT_FAST32: hid_t read FH5T_NATIVE_UINT_FAST32;
    property H5T_NATIVE_INT64: hid_t read FH5T_NATIVE_INT64;
    property H5T_NATIVE_UINT64: hid_t read FH5T_NATIVE_UINT64;
    property H5T_NATIVE_INT_LEAST64: hid_t read FH5T_NATIVE_INT_LEAST64;
    property H5T_NATIVE_UINT_LEAST64: hid_t read FH5T_NATIVE_UINT_LEAST64;
    property H5T_NATIVE_INT_FAST64: hid_t read FH5T_NATIVE_INT_FAST64;
    property H5T_NATIVE_UINT_FAST64: hid_t read FH5T_NATIVE_UINT_FAST64;
    property H5Tcreate: TH5Tcreate read FH5Tcreate;
    property H5Tcopy: TH5Tcopy read FH5Tcopy;
    property H5Tclose: TH5Tclose read FH5Tclose;
    property H5Tequal: TH5Tequal read FH5Tequal;
    property H5Tlock: TH5Tlock read FH5Tlock;
    property H5Tcommit2: TH5Tcommit2 read FH5Tcommit2;
    property H5Topen2: TH5Topen2 read FH5Topen2;
    property H5Tcommit_anon: TH5Tcommit_anon read FH5Tcommit_anon;
    property H5Tget_create_plist: TH5Tget_create_plist read FH5Tget_create_plist;
    property H5Tcommitted: TH5Tcommitted read FH5Tcommitted;
    property H5Tencode: TH5Tencode read FH5Tencode;
    property H5Tdecode: TH5Tdecode read FH5Tdecode;
    property H5Tflush: TH5Tflush read FH5Tflush;
    property H5Trefresh: TH5Trefresh read FH5Trefresh;
    property H5Tinsert: TH5Tinsert read FH5Tinsert;
    property H5Tpack: TH5Tpack read FH5Tpack;
    property H5Tenum_create: TH5Tenum_create read FH5Tenum_create;
    property H5Tenum_insert: TH5Tenum_insert read FH5Tenum_insert;
    property H5Tenum_nameof: TH5Tenum_nameof read FH5Tenum_nameof;
    property H5Tenum_valueof: TH5Tenum_valueof read FH5Tenum_valueof;
    property H5Tvlen_create: TH5Tvlen_create read FH5Tvlen_create;
    property H5Tarray_create2: TH5Tarray_create2 read FH5Tarray_create2;
    property H5Tget_array_ndims: TH5Tget_array_ndims read FH5Tget_array_ndims;
    property H5Tget_array_dims2: TH5Tget_array_dims2 read FH5Tget_array_dims2;
    property H5Tset_tag: TH5Tset_tag read FH5Tset_tag;
    property H5Tget_tag: TH5Tget_tag read FH5Tget_tag;
    property H5Tget_super: TH5Tget_super read FH5Tget_super;
    property H5Tget_class: TH5Tget_class read FH5Tget_class;
    property H5Tdetect_class: TH5Tdetect_class read FH5Tdetect_class;
    property H5Tget_size: TH5Tget_size read FH5Tget_size;
    property H5Tget_order: TH5Tget_order read FH5Tget_order;
    property H5Tget_precision: TH5Tget_precision read FH5Tget_precision;
    property H5Tget_offset: TH5Tget_offset read FH5Tget_offset;
    property H5Tget_pad: TH5Tget_pad read FH5Tget_pad;
    property H5Tget_sign: TH5Tget_sign read FH5Tget_sign;
    property H5Tget_fields: TH5Tget_fields read FH5Tget_fields;
    property H5Tget_ebias: TH5Tget_ebias read FH5Tget_ebias;
    property H5Tget_norm: TH5Tget_norm read FH5Tget_norm;
    property H5Tget_inpad: TH5Tget_inpad read FH5Tget_inpad;
    property H5Tget_strpad: TH5Tget_strpad read FH5Tget_strpad;
    property H5Tget_nmembers: TH5Tget_nmembers read FH5Tget_nmembers;
    property H5Tget_member_name: TH5Tget_member_name read FH5Tget_member_name;
    property H5Tget_member_index: TH5Tget_member_index read FH5Tget_member_index;
    property H5Tget_member_offset: TH5Tget_member_offset read FH5Tget_member_offset;
    property H5Tget_member_class: TH5Tget_member_class read FH5Tget_member_class;
    property H5Tget_member_type: TH5Tget_member_type read FH5Tget_member_type;
    property H5Tget_member_value: TH5Tget_member_value read FH5Tget_member_value;
    property H5Tget_cset: TH5Tget_cset read FH5Tget_cset;
    property H5Tis_variable_str: TH5Tis_variable_str read FH5Tis_variable_str;
    property H5Tget_native_type: TH5Tget_native_type read FH5Tget_native_type;
    property H5Tset_size: TH5Tset_size read FH5Tset_size;
    property H5Tset_order: TH5Tset_order read FH5Tset_order;
    property H5Tset_precision: TH5Tset_precision read FH5Tset_precision;
    property H5Tset_offset: TH5Tset_offset read FH5Tset_offset;
    property H5Tset_pad: TH5Tset_pad read FH5Tset_pad;
    property H5Tset_sign: TH5Tset_sign read FH5Tset_sign;
    property H5Tset_fields: TH5Tset_fields read FH5Tset_fields;
    property H5Tset_ebias: TH5Tset_ebias read FH5Tset_ebias;
    property H5Tset_norm: TH5Tset_norm read FH5Tset_norm;
    property H5Tset_inpad: TH5Tset_inpad read FH5Tset_inpad;
    property H5Tset_cset: TH5Tset_cset read FH5Tset_cset;
    property H5Tset_strpad: TH5Tset_strpad read FH5Tset_strpad;
    property H5Tregister: TH5Tregister read FH5Tregister;
    property H5Tunregister: TH5Tunregister read FH5Tunregister;
    property H5Tfind: TH5Tfind read FH5Tfind;
    property H5Tcompiler_conv: TH5Tcompiler_conv read FH5Tcompiler_conv;
    property H5Tconvert: TH5Tconvert read FH5Tconvert;
    property H5Dcreate2: TH5Dcreate2 read FH5Dcreate2;
    property H5Dcreate_anon: TH5Dcreate_anon read FH5Dcreate_anon;
    property H5Dopen2: TH5Dopen2 read FH5Dopen2;
    property H5Dclose: TH5Dclose read FH5Dclose;
    property H5Dget_space: TH5Dget_space read FH5Dget_space;
    property H5Dget_space_status: TH5Dget_space_status read FH5Dget_space_status;
    property H5Dget_type: TH5Dget_type read FH5Dget_type;
    property H5Dget_create_plist: TH5Dget_create_plist read FH5Dget_create_plist;
    property H5Dget_access_plist: TH5Dget_access_plist read FH5Dget_access_plist;
    property H5Dget_storage_size: TH5Dget_storage_size read FH5Dget_storage_size;
    property H5Dget_offset: TH5Dget_offset read FH5Dget_offset;
    property H5Dread: TH5Dread read FH5Dread;
    property H5Dwrite: TH5Dwrite read FH5Dwrite;
    property H5Diterate: TH5Diterate read FH5Diterate;
    property H5Dvlen_reclaim: TH5Dvlen_reclaim read FH5Dvlen_reclaim;
    property H5Dvlen_get_buf_size: TH5Dvlen_get_buf_size read FH5Dvlen_get_buf_size;
    property H5Dfill: TH5Dfill read FH5Dfill;
    property H5Dset_extent: TH5Dset_extent read FH5Dset_extent;
    property H5Dflush: TH5Dflush read FH5Dflush;
    property H5Drefresh: TH5Drefresh read FH5Drefresh;
    property H5Dscatter: TH5Dscatter read FH5Dscatter;
    property H5Dgather: TH5Dgather read FH5Dgather;
    property H5Ddebug: TH5Ddebug read FH5Ddebug;
    property H5Dformat_convert: TH5Dformat_convert read FH5Dformat_convert;
    property H5Dget_chunk_index_type: TH5Dget_chunk_index_type read FH5Dget_chunk_index_type;
    property H5E_ERR_CLS: hid_t read FH5E_ERR_CLS;
    property H5Eregister_class: TH5Eregister_class read FH5Eregister_class;
    property H5Eunregister_class: TH5Eunregister_class read FH5Eunregister_class;
    property H5Eclose_msg: TH5Eclose_msg read FH5Eclose_msg;
    property H5Ecreate_msg: TH5Ecreate_msg read FH5Ecreate_msg;
    property H5Ecreate_stack: TH5Ecreate_stack read FH5Ecreate_stack;
    property H5Eget_current_stack: TH5Eget_current_stack read FH5Eget_current_stack;
    property H5Eclose_stack: TH5Eclose_stack read FH5Eclose_stack;
    property H5Eget_class_name: TH5Eget_class_name read FH5Eget_class_name;
    property H5Eset_current_stack: TH5Eset_current_stack read FH5Eset_current_stack;
    // property H5Epush2: TH5Epush2 read H5Epush2;
    property H5Epop: TH5Epop read FH5Epop;
    property H5Eprint2: TH5Eprint2 read FH5Eprint2;
    property H5Ewalk2: TH5Ewalk2 read FH5Ewalk2;
    property H5Eget_auto2: TH5Eget_auto2 read FH5Eget_auto2;
    property H5Eset_auto2: TH5Eset_auto2 read FH5Eset_auto2;
    property H5Eclear2: TH5Eclear2 read FH5Eclear2;
    property H5Eauto_is_v2: TH5Eauto_is_v2 read FH5Eauto_is_v2;
    property H5Eget_msg: TH5Eget_msg read FH5Eget_msg;
    property H5Eget_num: TH5Eget_num read FH5Eget_num;
    property H5Screate: TH5Screate read FH5Screate;
    property H5Screate_simple: TH5Screate_simple read FH5Screate_simple;
    property H5Sset_extent_simple: TH5Sset_extent_simple read FH5Sset_extent_simple;
    property H5Scopy: TH5Scopy read FH5Scopy;
    property H5Sclose: TH5Sclose read FH5Sclose;
    property H5Sencode: TH5Sencode read FH5Sencode;
    property H5Sdecode: TH5Sdecode read FH5Sdecode;
    property H5Sget_simple_extent_npoints: TH5Sget_simple_extent_npoints read FH5Sget_simple_extent_npoints;
    property H5Sget_simple_extent_ndims: TH5Sget_simple_extent_ndims read FH5Sget_simple_extent_ndims;
    property H5Sget_simple_extent_dims: TH5Sget_simple_extent_dims read FH5Sget_simple_extent_dims;
    property H5Sis_simple: TH5Sis_simple read FH5Sis_simple;
    property H5Sget_select_npoints: TH5Sget_select_npoints read FH5Sget_select_npoints;
    property H5Sselect_hyperslab: TH5Sselect_hyperslab read FH5Sselect_hyperslab;
    property H5Sselect_elements: TH5Sselect_elements read FH5Sselect_elements;
    property H5Sget_simple_extent_type: TH5Sget_simple_extent_type read FH5Sget_simple_extent_type;
    property H5Sset_extent_none: TH5Sset_extent_none read FH5Sset_extent_none;
    property H5Sextent_copy: TH5Sextent_copy read FH5Sextent_copy;
    property H5Sextent_equal: TH5Sextent_equal read FH5Sextent_equal;
    property H5Sselect_all: TH5Sselect_all read FH5Sselect_all;
    property H5Sselect_none: TH5Sselect_none read FH5Sselect_none;
    property H5Soffset_simple: TH5Soffset_simple read FH5Soffset_simple;
    property H5Sselect_valid: TH5Sselect_valid read FH5Sselect_valid;
    property H5Sis_regular_hyperslab: TH5Sis_regular_hyperslab read FH5Sis_regular_hyperslab;
    property H5Sget_regular_hyperslab: TH5Sget_regular_hyperslab read FH5Sget_regular_hyperslab;
    property H5Sget_select_hyper_nblocks: TH5Sget_select_hyper_nblocks read FH5Sget_select_hyper_nblocks;
    property H5Sget_select_elem_npoints: TH5Sget_select_elem_npoints read FH5Sget_select_elem_npoints;
    property H5Sget_select_hyper_blocklist: TH5Sget_select_hyper_blocklist read FH5Sget_select_hyper_blocklist;
    property H5Sget_select_elem_pointlist: TH5Sget_select_elem_pointlist read FH5Sget_select_elem_pointlist;
    property H5Sget_select_bounds: TH5Sget_select_bounds read FH5Sget_select_bounds;
    property H5Sget_select_type: TH5Sget_select_type read FH5Sget_select_type;
    property H5Lmove: TH5Lmove read FH5Lmove;
    property H5Lcopy: TH5Lcopy read FH5Lcopy;
    property H5Lcreate_hard: TH5Lcreate_hard read FH5Lcreate_hard;
    property H5Lcreate_soft: TH5Lcreate_soft read FH5Lcreate_soft;
    property H5Ldelete: TH5Ldelete read FH5Ldelete;
    property H5Ldelete_by_idx: TH5Ldelete_by_idx read FH5Ldelete_by_idx;
    property H5Lget_val: TH5Lget_val read FH5Lget_val;
    property H5Lget_val_by_idx: TH5Lget_val_by_idx read FH5Lget_val_by_idx;
    property H5Lexists: TH5Lexists read FH5Lexists;
    property H5Lget_info: TH5Lget_info read FH5Lget_info;
    property H5Lget_info_by_idx: TH5Lget_info_by_idx read FH5Lget_info_by_idx;
    property H5Lget_name_by_idx: TH5Lget_name_by_idx read FH5Lget_name_by_idx;
    property H5Literate: TH5Literate read FH5Literate;
    property H5Literate_by_name: TH5Literate_by_name read FH5Literate_by_name;
    property H5Lvisit: TH5Lvisit read FH5Lvisit;
    property H5Lvisit_by_name: TH5Lvisit_by_name read FH5Lvisit_by_name;
    property H5Lcreate_ud: TH5Lcreate_ud read FH5Lcreate_ud;
    property H5Lregister: TH5Lregister read FH5Lregister;
    property H5Lunregister: TH5Lunregister read FH5Lunregister;
    property H5Lis_registered: TH5Lis_registered read FH5Lis_registered;
    property H5Lunpack_elink_val: TH5Lunpack_elink_val read FH5Lunpack_elink_val;
    property H5Lcreate_external: TH5Lcreate_external read FH5Lcreate_external;
    property H5Oopen: TH5Oopen read FH5Oopen;
    property H5Oopen_by_addr: TH5Oopen_by_addr read FH5Oopen_by_addr;
    property H5Oopen_by_idx: TH5Oopen_by_idx read FH5Oopen_by_idx;
    property H5Oexists_by_name: TH5Oexists_by_name read FH5Oexists_by_name;
    property H5Oget_info: TH5Oget_info read FH5Oget_info;
    property H5Oget_info_by_name: TH5Oget_info_by_name read FH5Oget_info_by_name;
    property H5Oget_info_by_idx: TH5Oget_info_by_idx read FH5Oget_info_by_idx;
    property H5Olink: TH5Olink read FH5Olink;
    property H5Oincr_refcount: TH5Oincr_refcount read FH5Oincr_refcount;
    property H5Odecr_refcount: TH5Odecr_refcount read FH5Odecr_refcount;
    property H5Ocopy: TH5Ocopy read FH5Ocopy;
    property H5Oset_comment: TH5Oset_comment read FH5Oset_comment;
    property H5Oset_comment_by_name: TH5Oset_comment_by_name read FH5Oset_comment_by_name;
    property H5Oget_comment: TH5Oget_comment read FH5Oget_comment;
    property H5Oget_comment_by_name: TH5Oget_comment_by_name read FH5Oget_comment_by_name;
    property H5Ovisit: TH5Ovisit read FH5Ovisit;
    property H5Ovisit_by_name: TH5Ovisit_by_name read FH5Ovisit_by_name;
    property H5Oclose: TH5Oclose read FH5Oclose;
    property H5Oflush: TH5Oflush read FH5Oflush;
    property H5Orefresh: TH5Orefresh read FH5Orefresh;
    property H5Odisable_mdc_flushes: TH5Odisable_mdc_flushes read FH5Odisable_mdc_flushes;
    property H5Oenable_mdc_flushes: TH5Oenable_mdc_flushes read FH5Oenable_mdc_flushes;
    property H5Oare_mdc_flushes_disabled: TH5Oare_mdc_flushes_disabled read FH5Oare_mdc_flushes_disabled;
    property H5Fis_hdf5: TH5Fis_hdf5 read FH5Fis_hdf5;
    property H5Fcreate: TH5Fcreate read FH5Fcreate;
    property H5Fopen: TH5Fopen read FH5Fopen;
    property H5Freopen: TH5Freopen read FH5Freopen;
    property H5Fflush: TH5Fflush read FH5Fflush;
    property H5Fclose: TH5Fclose read FH5Fclose;
    property H5Fget_create_plist: TH5Fget_create_plist read FH5Fget_create_plist;
    property H5Fget_access_plist: TH5Fget_access_plist read FH5Fget_access_plist;
    property H5Fget_intent: TH5Fget_intent read FH5Fget_intent;
    property H5Fget_obj_count: TH5Fget_obj_count read FH5Fget_obj_count;
    property H5Fget_obj_ids: TH5Fget_obj_ids read FH5Fget_obj_ids;
    property H5Fget_vfd_handle: TH5Fget_vfd_handle read FH5Fget_vfd_handle;
    property H5Fmount: TH5Fmount read FH5Fmount;
    property H5Funmount: TH5Funmount read FH5Funmount;
    property H5Fget_freespace: TH5Fget_freespace read FH5Fget_freespace;
    property H5Fget_filesize: TH5Fget_filesize read FH5Fget_filesize;
    property H5Fget_file_image: TH5Fget_file_image read FH5Fget_file_image;
    property H5Fget_mdc_config: TH5Fget_mdc_config read FH5Fget_mdc_config;
    property H5Fset_mdc_config: TH5Fset_mdc_config read FH5Fset_mdc_config;
    property H5Fget_mdc_hit_rate: TH5Fget_mdc_hit_rate read FH5Fget_mdc_hit_rate;
    property H5Fget_mdc_size: TH5Fget_mdc_size read FH5Fget_mdc_size;
    property H5Freset_mdc_hit_rate_stats: TH5Freset_mdc_hit_rate_stats read FH5Freset_mdc_hit_rate_stats;
    property H5Fget_name: TH5Fget_name read FH5Fget_name;
    property H5Fget_info2: TH5Fget_info2 read FH5Fget_info2;
    property H5Fget_metadata_read_retry_info: TH5Fget_metadata_read_retry_info read FH5Fget_metadata_read_retry_info;
    property H5Fstart_swmr_write: TH5Fstart_swmr_write read FH5Fstart_swmr_write;
    property H5Fget_free_sections: TH5Fget_free_sections read FH5Fget_free_sections;
    property H5Fclear_elink_file_cache: TH5Fclear_elink_file_cache read FH5Fclear_elink_file_cache;
    property H5Fstart_mdc_logging: TH5Fstart_mdc_logging read FH5Fstart_mdc_logging;
    property H5Fstop_mdc_logging: TH5Fstop_mdc_logging read FH5Fstop_mdc_logging;
    property H5Fget_mdc_logging_status: TH5Fget_mdc_logging_status read FH5Fget_mdc_logging_status;
    property H5Fformat_convert: TH5Fformat_convert read FH5Fformat_convert;
    property H5Acreate2: TH5Acreate2 read FH5Acreate2;
    property H5Acreate_by_name: TH5Acreate_by_name read FH5Acreate_by_name;
    property H5Aopen: TH5Aopen read FH5Aopen;
    property H5Aopen_by_name: TH5Aopen_by_name read FH5Aopen_by_name;
    property H5Aopen_by_idx: TH5Aopen_by_idx read FH5Aopen_by_idx;
    property H5Awrite: TH5Awrite read FH5Awrite;
    property H5Aread: TH5Aread read FH5Aread;
    property H5Aclose: TH5Aclose read FH5Aclose;
    property H5Aget_space: TH5Aget_space read FH5Aget_space;
    property H5Aget_type: TH5Aget_type read FH5Aget_type;
    property H5Aget_create_plist: TH5Aget_create_plist read FH5Aget_create_plist;
    property H5Aget_name: TH5Aget_name read FH5Aget_name;
    property H5Aget_name_by_idx: TH5Aget_name_by_idx read FH5Aget_name_by_idx;
    property H5Aget_storage_size: TH5Aget_storage_size read FH5Aget_storage_size;
    property H5Aget_info: TH5Aget_info read FH5Aget_info;
    property H5Aget_info_by_name: TH5Aget_info_by_name read FH5Aget_info_by_name;
    property H5Aget_info_by_idx: TH5Aget_info_by_idx read FH5Aget_info_by_idx;
    property H5Arename: TH5Arename read FH5Arename;
    property H5Arename_by_name: TH5Arename_by_name read FH5Arename_by_name;
    property H5Aiterate2: TH5Aiterate2 read FH5Aiterate2;
    property H5Aiterate_by_name: TH5Aiterate_by_name read FH5Aiterate_by_name;
    property H5Adelete: TH5Adelete read FH5Adelete;
    property H5Adelete_by_name: TH5Adelete_by_name read FH5Adelete_by_name;
    property H5Adelete_by_idx: TH5Adelete_by_idx read FH5Adelete_by_idx;
    property H5Aexists: TH5Aexists read FH5Aexists;
    property H5Aexists_by_name: TH5Aexists_by_name read FH5Aexists_by_name;
    property H5FDregister: TH5FDregister read FH5FDregister;
    property H5FDunregister: TH5FDunregister read FH5FDunregister;
    property H5FDopen: TH5FDopen read FH5FDopen;
    property H5FDclose: TH5FDclose read FH5FDclose;
    property H5FDcmp: TH5FDcmp read FH5FDcmp;
    property H5FDquery: TH5FDquery read FH5FDquery;
    property H5FDalloc: TH5FDalloc read FH5FDalloc;
    property H5FDfree: TH5FDfree read FH5FDfree;
    property H5FDget_eoa: TH5FDget_eoa read FH5FDget_eoa;
    property H5FDset_eoa: TH5FDset_eoa read FH5FDset_eoa;
    property H5FDget_eof: TH5FDget_eof read FH5FDget_eof;
    property H5FDget_vfd_handle: TH5FDget_vfd_handle read FH5FDget_vfd_handle;
    property H5FDread: TH5FDread read FH5FDread;
    property H5FDwrite: TH5FDwrite read FH5FDwrite;
    property H5FDflush: TH5FDflush read FH5FDflush;
    property H5FDtruncate: TH5FDtruncate read FH5FDtruncate;
    property H5FDlock: TH5FDlock read FH5FDlock;
    property H5FDunlock: TH5FDunlock read FH5FDunlock;
    property H5Gcreate2: TH5Gcreate2 read FH5Gcreate2;
    property H5Gcreate_anon: TH5Gcreate_anon read FH5Gcreate_anon;
    property H5Gopen2: TH5Gopen2 read FH5Gopen2;
    property H5Gget_create_plist: TH5Gget_create_plist read FH5Gget_create_plist;
    property H5Gget_info: TH5Gget_info read FH5Gget_info;
    property H5Gget_info_by_name: TH5Gget_info_by_name read FH5Gget_info_by_name;
    property H5Gget_info_by_idx: TH5Gget_info_by_idx read FH5Gget_info_by_idx;
    property H5Gclose: TH5Gclose read FH5Gclose;
    property H5Gflush: TH5Gflush read FH5Gflush;
    property H5Grefresh: TH5Grefresh read FH5Grefresh;
    property H5Rcreate: TH5Rcreate read FH5Rcreate;
    property H5Rdereference2: TH5Rdereference2 read FH5Rdereference2;
    property H5Rget_region: TH5Rget_region read FH5Rget_region;
    property H5Rget_obj_type2: TH5Rget_obj_type2 read FH5Rget_obj_type2;
    property H5Rget_name: TH5Rget_name read FH5Rget_name;
    property H5P_ROOT: hid_t read FH5P_CLS_ROOT_ID;
    property H5P_OBJECT_CREATE: hid_t read FH5P_CLS_OBJECT_CREATE_ID;
    property H5P_FILE_CREATE: hid_t read FH5P_CLS_FILE_CREATE_ID;
    property H5P_FILE_ACCESS: hid_t read FH5P_CLS_FILE_ACCESS_ID;
    property H5P_DATASET_CREATE: hid_t read FH5P_CLS_DATASET_CREATE_ID;
    property H5P_DATASET_ACCESS: hid_t read FH5P_CLS_DATASET_ACCESS_ID;
    property H5P_DATASET_XFER: hid_t read FH5P_CLS_DATASET_XFER_ID;
    property H5P_FILE_MOUNT: hid_t read FH5P_CLS_FILE_MOUNT_ID;
    property H5P_GROUP_CREATE: hid_t read FH5P_CLS_GROUP_CREATE_ID;
    property H5P_GROUP_ACCESS: hid_t read FH5P_CLS_GROUP_ACCESS_ID;
    property H5P_DATATYPE_CREATE: hid_t read FH5P_CLS_DATATYPE_CREATE_ID;
    property H5P_DATATYPE_ACCESS: hid_t read FH5P_CLS_DATATYPE_ACCESS_ID;
    property H5P_STRING_CREATE: hid_t read FH5P_CLS_STRING_CREATE_ID;
    property H5P_ATTRIBUTE_CREATE: hid_t read FH5P_CLS_ATTRIBUTE_CREATE_ID;
    property H5P_ATTRIBUTE_ACCESS: hid_t read FH5P_CLS_ATTRIBUTE_ACCESS_ID;
    property H5P_OBJECT_COPY: hid_t read FH5P_CLS_OBJECT_COPY_ID;
    property H5P_LINK_CREATE: hid_t read FH5P_CLS_LINK_CREATE_ID;
    property H5P_LINK_ACCESS: hid_t read FH5P_CLS_LINK_ACCESS_ID;
    property H5P_FILE_CREATE_DEFAULT: hid_t read FH5P_LST_FILE_CREATE_ID;
    property H5P_FILE_ACCESS_DEFAULT: hid_t read FH5P_LST_FILE_ACCESS_ID;
    property H5P_DATASET_CREATE_DEFAULT: hid_t read FH5P_LST_DATASET_CREATE_ID;
    property H5P_DATASET_ACCESS_DEFAULT: hid_t read FH5P_LST_DATASET_ACCESS_ID;
    property H5P_DATASET_XFER_DEFAULT: hid_t read FH5P_LST_DATASET_XFER_ID;
    property H5P_FILE_MOUNT_DEFAULT: hid_t read FH5P_LST_FILE_MOUNT_ID;
    property H5P_GROUP_CREATE_DEFAULT: hid_t read FH5P_LST_GROUP_CREATE_ID;
    property H5P_GROUP_ACCESS_DEFAULT: hid_t read FH5P_LST_GROUP_ACCESS_ID;
    property H5P_DATATYPE_CREATE_DEFAULT: hid_t read FH5P_LST_DATATYPE_CREATE_ID;
    property H5P_DATATYPE_ACCESS_DEFAULT: hid_t read FH5P_LST_DATATYPE_ACCESS_ID;
    property H5P_ATTRIBUTE_CREATE_DEFAULT: hid_t read FH5P_LST_ATTRIBUTE_CREATE_ID;
    property H5P_ATTRIBUTE_ACCESS_DEFAULT: hid_t read FH5P_LST_ATTRIBUTE_ACCESS_ID;
    property H5P_OBJECT_COPY_DEFAULT: hid_t read FH5P_LST_OBJECT_COPY_ID;
    property H5P_LINK_CREATE_DEFAULT: hid_t read FH5P_LST_LINK_CREATE_ID;
    property H5P_LINK_ACCESS_DEFAULT: hid_t read FH5P_LST_LINK_ACCESS_ID;
    property H5Pcreate_class: TH5Pcreate_class read FH5Pcreate_class;
    property H5Pget_class_name: TH5Pget_class_name read FH5Pget_class_name;
    property H5Pcreate: TH5Pcreate read FH5Pcreate;
    property H5Pregister2: TH5Pregister2 read FH5Pregister2;
    property H5Pinsert2: TH5Pinsert2 read FH5Pinsert2;
    property H5Pset: TH5Pset read FH5Pset;
    property H5Pexist: TH5Pexist read FH5Pexist;
    property H5Pencode: TH5Pencode read FH5Pencode;
    property H5Pdecode: TH5Pdecode read FH5Pdecode;
    property H5Pget_size: TH5Pget_size read FH5Pget_size;
    property H5Pget_nprops: TH5Pget_nprops read FH5Pget_nprops;
    property H5Pget_class: TH5Pget_class read FH5Pget_class;
    property H5Pget_class_parent: TH5Pget_class_parent read FH5Pget_class_parent;
    property H5Pget: TH5Pget read FH5Pget;
    property H5Pequal: TH5Pequal read FH5Pequal;
    property H5Pisa_class: TH5Pisa_class read FH5Pisa_class;
    property H5Piterate: TH5Piterate read FH5Piterate;
    property H5Pcopy_prop: TH5Pcopy_prop read FH5Pcopy_prop;
    property H5Premove: TH5Premove read FH5Premove;
    property H5Punregister: TH5Punregister read FH5Punregister;
    property H5Pclose_class: TH5Pclose_class read FH5Pclose_class;
    property H5Pclose: TH5Pclose read FH5Pclose;
    property H5Pcopy: TH5Pcopy read FH5Pcopy;
    property H5Pset_attr_phase_change: TH5Pset_attr_phase_change read FH5Pset_attr_phase_change;
    property H5Pget_attr_phase_change: TH5Pget_attr_phase_change read FH5Pget_attr_phase_change;
    property H5Pset_attr_creation_order: TH5Pset_attr_creation_order read FH5Pset_attr_creation_order;
    property H5Pget_attr_creation_order: TH5Pget_attr_creation_order read FH5Pget_attr_creation_order;
    property H5Pset_obj_track_times: TH5Pset_obj_track_times read FH5Pset_obj_track_times;
    property H5Pget_obj_track_times: TH5Pget_obj_track_times read FH5Pget_obj_track_times;
    property H5Pmodify_filter: TH5Pmodify_filter read FH5Pmodify_filter;
    property H5Pset_filter: TH5Pset_filter read FH5Pset_filter;
    property H5Pget_nfilters: TH5Pget_nfilters read FH5Pget_nfilters;
    property H5Pget_filter2: TH5Pget_filter2 read FH5Pget_filter2;
    property H5Pget_filter_by_id2: TH5Pget_filter_by_id2 read FH5Pget_filter_by_id2;
    property H5Pall_filters_avail: TH5Pall_filters_avail read FH5Pall_filters_avail;
    property H5Premove_filter: TH5Premove_filter read FH5Premove_filter;
    property H5Pset_deflate: TH5Pset_deflate read FH5Pset_deflate;
    property H5Pset_fletcher32: TH5Pset_fletcher32 read FH5Pset_fletcher32;
    property H5Pset_userblock: TH5Pset_userblock read FH5Pset_userblock;
    property H5Pget_userblock: TH5Pget_userblock read FH5Pget_userblock;
    property H5Pset_sizes: TH5Pset_sizes read FH5Pset_sizes;
    property H5Pget_sizes: TH5Pget_sizes read FH5Pget_sizes;
    property H5Pset_sym_k: TH5Pset_sym_k read FH5Pset_sym_k;
    property H5Pget_sym_k: TH5Pget_sym_k read FH5Pget_sym_k;
    property H5Pset_istore_k: TH5Pset_istore_k read FH5Pset_istore_k;
    property H5Pget_istore_k: TH5Pget_istore_k read FH5Pget_istore_k;
    property H5Pset_shared_mesg_nindexes: TH5Pset_shared_mesg_nindexes read FH5Pset_shared_mesg_nindexes;
    property H5Pget_shared_mesg_nindexes: TH5Pget_shared_mesg_nindexes read FH5Pget_shared_mesg_nindexes;
    property H5Pset_shared_mesg_index: TH5Pset_shared_mesg_index read FH5Pset_shared_mesg_index;
    property H5Pget_shared_mesg_index: TH5Pget_shared_mesg_index read FH5Pget_shared_mesg_index;
    property H5Pset_shared_mesg_phase_change: TH5Pset_shared_mesg_phase_change read FH5Pset_shared_mesg_phase_change;
    property H5Pget_shared_mesg_phase_change: TH5Pget_shared_mesg_phase_change read FH5Pget_shared_mesg_phase_change;
    property H5Pset_file_space: TH5Pset_file_space read FH5Pset_file_space;
    property H5Pget_file_space: TH5Pget_file_space read FH5Pget_file_space;
    property H5Pset_alignment: TH5Pset_alignment read FH5Pset_alignment;
    property H5Pget_alignment: TH5Pget_alignment read FH5Pget_alignment;
    property H5Pset_driver: TH5Pset_driver read FH5Pset_driver;
    property H5Pget_driver: TH5Pget_driver read FH5Pget_driver;
    property H5Pget_driver_info: TH5Pget_driver_info read FH5Pget_driver_info;
    property H5Pset_family_offset: TH5Pset_family_offset read FH5Pset_family_offset;
    property H5Pget_family_offset: TH5Pget_family_offset read FH5Pget_family_offset;
    property H5Pset_multi_type: TH5Pset_multi_type read FH5Pset_multi_type;
    property H5Pget_multi_type: TH5Pget_multi_type read FH5Pget_multi_type;
    property H5Pset_cache: TH5Pset_cache read FH5Pset_cache;
    property H5Pget_cache: TH5Pget_cache read FH5Pget_cache;
    property H5Pset_mdc_config: TH5Pset_mdc_config read FH5Pset_mdc_config;
    property H5Pget_mdc_config: TH5Pget_mdc_config read FH5Pget_mdc_config;
    property H5Pset_gc_references: TH5Pset_gc_references read FH5Pset_gc_references;
    property H5Pget_gc_references: TH5Pget_gc_references read FH5Pget_gc_references;
    property H5Pset_fclose_degree: TH5Pset_fclose_degree read FH5Pset_fclose_degree;
    property H5Pget_fclose_degree: TH5Pget_fclose_degree read FH5Pget_fclose_degree;
    property H5Pset_meta_block_size: TH5Pset_meta_block_size read FH5Pset_meta_block_size;
    property H5Pget_meta_block_size: TH5Pget_meta_block_size read FH5Pget_meta_block_size;
    property H5Pset_sieve_buf_size: TH5Pset_sieve_buf_size read FH5Pset_sieve_buf_size;
    property H5Pget_sieve_buf_size: TH5Pget_sieve_buf_size read FH5Pget_sieve_buf_size;
    property H5Pset_small_data_block_size: TH5Pset_small_data_block_size read FH5Pset_small_data_block_size;
    property H5Pget_small_data_block_size: TH5Pget_small_data_block_size read FH5Pget_small_data_block_size;
    property H5Pset_libver_bounds: TH5Pset_libver_bounds read FH5Pset_libver_bounds;
    property H5Pget_libver_bounds: TH5Pget_libver_bounds read FH5Pget_libver_bounds;
    property H5Pset_elink_file_cache_size: TH5Pset_elink_file_cache_size read FH5Pset_elink_file_cache_size;
    property H5Pget_elink_file_cache_size: TH5Pget_elink_file_cache_size read FH5Pget_elink_file_cache_size;
    property H5Pset_file_image: TH5Pset_file_image read FH5Pset_file_image;
    property H5Pget_file_image: TH5Pget_file_image read FH5Pget_file_image;
    property H5Pset_file_image_callbacks: TH5Pset_file_image_callbacks read FH5Pset_file_image_callbacks;
    property H5Pget_file_image_callbacks: TH5Pget_file_image_callbacks read FH5Pget_file_image_callbacks;
    property H5Pset_core_write_tracking: TH5Pset_core_write_tracking read FH5Pset_core_write_tracking;
    property H5Pget_core_write_tracking: TH5Pget_core_write_tracking read FH5Pget_core_write_tracking;
    property H5Pset_metadata_read_attempts: TH5Pset_metadata_read_attempts read FH5Pset_metadata_read_attempts;
    property H5Pget_metadata_read_attempts: TH5Pget_metadata_read_attempts read FH5Pget_metadata_read_attempts;
    property H5Pset_object_flush_cb: TH5Pset_object_flush_cb read FH5Pset_object_flush_cb;
    property H5Pget_object_flush_cb: TH5Pget_object_flush_cb read FH5Pget_object_flush_cb;
    property H5Pset_mdc_log_options: TH5Pset_mdc_log_options read FH5Pset_mdc_log_options;
    property H5Pget_mdc_log_options: TH5Pget_mdc_log_options read FH5Pget_mdc_log_options;
    property H5Pset_layout: TH5Pset_layout read FH5Pset_layout;
    property H5Pget_layout: TH5Pget_layout read FH5Pget_layout;
    property H5Pset_chunk: TH5Pset_chunk read FH5Pset_chunk;
    property H5Pget_chunk: TH5Pget_chunk read FH5Pget_chunk;
    property H5Pset_virtual: TH5Pset_virtual read FH5Pset_virtual;
    property H5Pget_virtual_count: TH5Pget_virtual_count read FH5Pget_virtual_count;
    property H5Pget_virtual_vspace: TH5Pget_virtual_vspace read FH5Pget_virtual_vspace;
    property H5Pget_virtual_srcspace: TH5Pget_virtual_srcspace read FH5Pget_virtual_srcspace;
    property H5Pget_virtual_filename: TH5Pget_virtual_filename read FH5Pget_virtual_filename;
    property H5Pget_virtual_dsetname: TH5Pget_virtual_dsetname read FH5Pget_virtual_dsetname;
    property H5Pset_external: TH5Pset_external read FH5Pset_external;
    property H5Pset_chunk_opts: TH5Pset_chunk_opts read FH5Pset_chunk_opts;
    property H5Pget_chunk_opts: TH5Pget_chunk_opts read FH5Pget_chunk_opts;
    property H5Pget_external_count: TH5Pget_external_count read FH5Pget_external_count;
    property H5Pget_external: TH5Pget_external read FH5Pget_external;
    property H5Pset_szip: TH5Pset_szip read FH5Pset_szip;
    property H5Pset_shuffle: TH5Pset_shuffle read FH5Pset_shuffle;
    property H5Pset_nbit: TH5Pset_nbit read FH5Pset_nbit;
    property H5Pset_scaleoffset: TH5Pset_scaleoffset read FH5Pset_scaleoffset;
    property H5Pset_fill_value: TH5Pset_fill_value read FH5Pset_fill_value;
    property H5Pget_fill_value: TH5Pget_fill_value read FH5Pget_fill_value;
    property H5Pfill_value_defined: TH5Pfill_value_defined read FH5Pfill_value_defined;
    property H5Pset_alloc_time: TH5Pset_alloc_time read FH5Pset_alloc_time;
    property H5Pget_alloc_time: TH5Pget_alloc_time read FH5Pget_alloc_time;
    property H5Pset_fill_time: TH5Pset_fill_time read FH5Pset_fill_time;
    property H5Pget_fill_time: TH5Pget_fill_time read FH5Pget_fill_time;
    property H5Pset_chunk_cache: TH5Pset_chunk_cache read FH5Pset_chunk_cache;
    property H5Pget_chunk_cache: TH5Pget_chunk_cache read FH5Pget_chunk_cache;
    property H5Pset_virtual_view: TH5Pset_virtual_view read FH5Pset_virtual_view;
    property H5Pget_virtual_view: TH5Pget_virtual_view read FH5Pget_virtual_view;
    property H5Pset_virtual_printf_gap: TH5Pset_virtual_printf_gap read FH5Pset_virtual_printf_gap;
    property H5Pget_virtual_printf_gap: TH5Pget_virtual_printf_gap read FH5Pget_virtual_printf_gap;
    property H5Pset_append_flush: TH5Pset_append_flush read FH5Pset_append_flush;
    property H5Pget_append_flush: TH5Pget_append_flush read FH5Pget_append_flush;
    property H5Pset_efile_prefix: TH5Pset_efile_prefix read FH5Pset_efile_prefix;
    property H5Pget_efile_prefix: TH5Pget_efile_prefix read FH5Pget_efile_prefix;
    property H5Pset_data_transform: TH5Pset_data_transform read FH5Pset_data_transform;
    property H5Pget_data_transform: TH5Pget_data_transform read FH5Pget_data_transform;
    property H5Pset_buffer: TH5Pset_buffer read FH5Pset_buffer;
    property H5Pget_buffer: TH5Pget_buffer read FH5Pget_buffer;
    property H5Pset_preserve: TH5Pset_preserve read FH5Pset_preserve;
    property H5Pget_preserve: TH5Pget_preserve read FH5Pget_preserve;
    property H5Pset_edc_check: TH5Pset_edc_check read FH5Pset_edc_check;
    property H5Pget_edc_check: TH5Pget_edc_check read FH5Pget_edc_check;
    property H5Pset_filter_callback: TH5Pset_filter_callback read FH5Pset_filter_callback;
    property H5Pset_btree_ratios: TH5Pset_btree_ratios read FH5Pset_btree_ratios;
    property H5Pget_btree_ratios: TH5Pget_btree_ratios read FH5Pget_btree_ratios;
    property H5Pset_vlen_mem_manager: TH5Pset_vlen_mem_manager read FH5Pset_vlen_mem_manager;
    property H5Pget_vlen_mem_manager: TH5Pget_vlen_mem_manager read FH5Pget_vlen_mem_manager;
    property H5Pset_hyper_vector_size: TH5Pset_hyper_vector_size read FH5Pset_hyper_vector_size;
    property H5Pget_hyper_vector_size: TH5Pget_hyper_vector_size read FH5Pget_hyper_vector_size;
    property H5Pset_type_conv_cb: TH5Pset_type_conv_cb read FH5Pset_type_conv_cb;
    property H5Pget_type_conv_cb: TH5Pget_type_conv_cb read FH5Pget_type_conv_cb;
    property H5Pset_create_intermediate_group: TH5Pset_create_intermediate_group read FH5Pset_create_intermediate_group;
    property H5Pget_create_intermediate_group: TH5Pget_create_intermediate_group read FH5Pget_create_intermediate_group;
    property H5Pset_local_heap_size_hint: TH5Pset_local_heap_size_hint read FH5Pset_local_heap_size_hint;
    property H5Pget_local_heap_size_hint: TH5Pget_local_heap_size_hint read FH5Pget_local_heap_size_hint;
    property H5Pset_link_phase_change: TH5Pset_link_phase_change read FH5Pset_link_phase_change;
    property H5Pget_link_phase_change: TH5Pget_link_phase_change read FH5Pget_link_phase_change;
    property H5Pset_est_link_info: TH5Pset_est_link_info read FH5Pset_est_link_info;
    property H5Pget_est_link_info: TH5Pget_est_link_info read FH5Pget_est_link_info;
    property H5Pset_link_creation_order: TH5Pset_link_creation_order read FH5Pset_link_creation_order;
    property H5Pget_link_creation_order: TH5Pget_link_creation_order read FH5Pget_link_creation_order;
    property H5Pset_char_encoding: TH5Pset_char_encoding read FH5Pset_char_encoding;
    property H5Pget_char_encoding: TH5Pget_char_encoding read FH5Pget_char_encoding;
    property H5Pset_nlinks: TH5Pset_nlinks read FH5Pset_nlinks;
    property H5Pget_nlinks: TH5Pget_nlinks read FH5Pget_nlinks;
    property H5Pset_elink_prefix: TH5Pset_elink_prefix read FH5Pset_elink_prefix;
    property H5Pget_elink_prefix: TH5Pget_elink_prefix read FH5Pget_elink_prefix;
    property H5Pget_elink_fapl: TH5Pget_elink_fapl read FH5Pget_elink_fapl;
    property H5Pset_elink_fapl: TH5Pset_elink_fapl read FH5Pset_elink_fapl;
    property H5Pset_elink_acc_flags: TH5Pset_elink_acc_flags read FH5Pset_elink_acc_flags;
    property H5Pget_elink_acc_flags: TH5Pget_elink_acc_flags read FH5Pget_elink_acc_flags;
    property H5Pset_elink_cb: TH5Pset_elink_cb read FH5Pset_elink_cb;
    property H5Pget_elink_cb: TH5Pget_elink_cb read FH5Pget_elink_cb;
    property H5Pset_copy_object: TH5Pset_copy_object read FH5Pset_copy_object;
    property H5Pget_copy_object: TH5Pget_copy_object read FH5Pget_copy_object;
    property H5Padd_merge_committed_dtype_path: TH5Padd_merge_committed_dtype_path read FH5Padd_merge_committed_dtype_path;
    property H5Pfree_merge_committed_dtype_paths: TH5Pfree_merge_committed_dtype_paths read FH5Pfree_merge_committed_dtype_paths;
    property H5Pset_mcdt_search_cb: TH5Pset_mcdt_search_cb read FH5Pset_mcdt_search_cb;
    property H5Pget_mcdt_search_cb: TH5Pget_mcdt_search_cb read FH5Pget_mcdt_search_cb;

    property Handle: THandle read FHandle;
    function IsValid: Boolean;
  end;

implementation

{ THDF5Dll }
constructor THDF5Dll.Create(APath: string);

  function GetDllProc(AModule: THandle; AName: string): Pointer;
  begin
    Result := GetProcAddress(AModule, PChar(AName));
    Assert(Assigned(Result));
  end;

begin
  inherited Create;
  FHandle := LoadLibrary(PChar(APath));

  @FH5open := GetDllProc(FHandle, 'H5open');
  @FH5close := GetDllProc(FHandle, 'H5close');
  @FH5dont_atexit := GetDllProc(FHandle, 'H5dont_atexit');
  @FH5garbage_collect := GetDllProc(FHandle, 'H5garbage_collect');
  @FH5set_free_list_limits := GetDllProc(FHandle, 'H5set_free_list_limits');
  @FH5get_libversion := GetDllProc(FHandle, 'H5get_libversion');
  @FH5check_version := GetDllProc(FHandle, 'H5check_version');
  @FH5is_library_threadsafe := GetDllProc(FHandle, 'H5is_library_threadsafe');
  @FH5free_memory := GetDllProc(FHandle, 'H5free_memory');
  @FH5allocate_memory := GetDllProc(FHandle, 'H5allocate_memory');
  @FH5resize_memory := GetDllProc(FHandle, 'H5resize_memory');
  @FH5Iregister := GetDllProc(FHandle, 'H5Iregister');
  @FH5Iobject_verify := GetDllProc(FHandle, 'H5Iobject_verify');
  @FH5Iremove_verify := GetDllProc(FHandle, 'H5Iremove_verify');
  @FH5Iget_type := GetDllProc(FHandle, 'H5Iget_type');
  @FH5Iget_file_id := GetDllProc(FHandle, 'H5Iget_file_id');
  @FH5Iget_name := GetDllProc(FHandle, 'H5Iget_name');
  @FH5Iinc_ref := GetDllProc(FHandle, 'H5Iinc_ref');
  @FH5Idec_ref := GetDllProc(FHandle, 'H5Idec_ref');
  @FH5Iget_ref := GetDllProc(FHandle, 'H5Iget_ref');
  @FH5Iregister_type := GetDllProc(FHandle, 'H5Iregister_type');
  @FH5Iclear_type := GetDllProc(FHandle, 'H5Iclear_type');
  @FH5Idestroy_type := GetDllProc(FHandle, 'H5Idestroy_type');
  @FH5Iinc_type_ref := GetDllProc(FHandle, 'H5Iinc_type_ref');
  @FH5Idec_type_ref := GetDllProc(FHandle, 'H5Idec_type_ref');
  @FH5Iget_type_ref := GetDllProc(FHandle, 'H5Iget_type_ref');
  @FH5Isearch := GetDllProc(FHandle, 'H5Isearch');
  @FH5Inmembers := GetDllProc(FHandle, 'H5Inmembers');
  @FH5Itype_exists := GetDllProc(FHandle, 'H5Itype_exists');
  @FH5Iis_valid := GetDllProc(FHandle, 'H5Iis_valid');
  @FH5Zregister := GetDllProc(FHandle, 'H5Zregister');
  @FH5Zunregister := GetDllProc(FHandle, 'H5Zunregister');
  @FH5Zfilter_avail := GetDllProc(FHandle, 'H5Zfilter_avail');
  @FH5Zget_filter_info := GetDllProc(FHandle, 'H5Zget_filter_info');
  @FH5PLset_loading_state := GetDllProc(FHandle, 'H5PLset_loading_state');
  @FH5PLget_loading_state := GetDllProc(FHandle, 'H5PLget_loading_state');
  @FH5Tcreate := GetDllProc(FHandle, 'H5Tcreate');
  @FH5Tcopy := GetDllProc(FHandle, 'H5Tcopy');
  @FH5Tclose := GetDllProc(FHandle, 'H5Tclose');
  @FH5Tequal := GetDllProc(FHandle, 'H5Tequal');
  @FH5Tlock := GetDllProc(FHandle, 'H5Tlock');
  @FH5Tcommit2 := GetDllProc(FHandle, 'H5Tcommit2');
  @FH5Topen2 := GetDllProc(FHandle, 'H5Topen2');
  @FH5Tcommit_anon := GetDllProc(FHandle, 'H5Tcommit_anon');
  @FH5Tget_create_plist := GetDllProc(FHandle, 'H5Tget_create_plist');
  @FH5Tcommitted := GetDllProc(FHandle, 'H5Tcommitted');
  @FH5Tencode := GetDllProc(FHandle, 'H5Tencode');
  @FH5Tdecode := GetDllProc(FHandle, 'H5Tdecode');
  @FH5Tflush := GetDllProc(FHandle, 'H5Tflush');
  @FH5Trefresh := GetDllProc(FHandle, 'H5Trefresh');
  @FH5Tinsert := GetDllProc(FHandle, 'H5Tinsert');
  @FH5Tpack := GetDllProc(FHandle, 'H5Tpack');
  @FH5Tenum_create := GetDllProc(FHandle, 'H5Tenum_create');
  @FH5Tenum_insert := GetDllProc(FHandle, 'H5Tenum_insert');
  @FH5Tenum_nameof := GetDllProc(FHandle, 'H5Tenum_nameof');
  @FH5Tenum_valueof := GetDllProc(FHandle, 'H5Tenum_valueof');
  @FH5Tvlen_create := GetDllProc(FHandle, 'H5Tvlen_create');
  @FH5Tarray_create2 := GetDllProc(FHandle, 'H5Tarray_create2');
  @FH5Tget_array_ndims := GetDllProc(FHandle, 'H5Tget_array_ndims');
  @FH5Tget_array_dims2 := GetDllProc(FHandle, 'H5Tget_array_dims2');
  @FH5Tset_tag := GetDllProc(FHandle, 'H5Tset_tag');
  @FH5Tget_tag := GetDllProc(FHandle, 'H5Tget_tag');
  @FH5Tget_super := GetDllProc(FHandle, 'H5Tget_super');
  @FH5Tget_class := GetDllProc(FHandle, 'H5Tget_class');
  @FH5Tdetect_class := GetDllProc(FHandle, 'H5Tdetect_class');
  @FH5Tget_size := GetDllProc(FHandle, 'H5Tget_size');
  @FH5Tget_order := GetDllProc(FHandle, 'H5Tget_order');
  @FH5Tget_precision := GetDllProc(FHandle, 'H5Tget_precision');
  @FH5Tget_offset := GetDllProc(FHandle, 'H5Tget_offset');
  @FH5Tget_pad := GetDllProc(FHandle, 'H5Tget_pad');
  @FH5Tget_sign := GetDllProc(FHandle, 'H5Tget_sign');
  @FH5Tget_fields := GetDllProc(FHandle, 'H5Tget_fields');
  @FH5Tget_ebias := GetDllProc(FHandle, 'H5Tget_ebias');
  @FH5Tget_norm := GetDllProc(FHandle, 'H5Tget_norm');
  @FH5Tget_inpad := GetDllProc(FHandle, 'H5Tget_inpad');
  @FH5Tget_strpad := GetDllProc(FHandle, 'H5Tget_strpad');
  @FH5Tget_nmembers := GetDllProc(FHandle, 'H5Tget_nmembers');
  @FH5Tget_member_name := GetDllProc(FHandle, 'H5Tget_member_name');
  @FH5Tget_member_index := GetDllProc(FHandle, 'H5Tget_member_index');
  @FH5Tget_member_offset := GetDllProc(FHandle, 'H5Tget_member_offset');
  @FH5Tget_member_class := GetDllProc(FHandle, 'H5Tget_member_class');
  @FH5Tget_member_type := GetDllProc(FHandle, 'H5Tget_member_type');
  @FH5Tget_member_value := GetDllProc(FHandle, 'H5Tget_member_value');
  @FH5Tget_cset := GetDllProc(FHandle, 'H5Tget_cset');
  @FH5Tis_variable_str := GetDllProc(FHandle, 'H5Tis_variable_str');
  @FH5Tget_native_type := GetDllProc(FHandle, 'H5Tget_native_type');
  @FH5Tset_size := GetDllProc(FHandle, 'H5Tset_size');
  @FH5Tset_order := GetDllProc(FHandle, 'H5Tset_order');
  @FH5Tset_precision := GetDllProc(FHandle, 'H5Tset_precision');
  @FH5Tset_offset := GetDllProc(FHandle, 'H5Tset_offset');
  @FH5Tset_pad := GetDllProc(FHandle, 'H5Tset_pad');
  @FH5Tset_sign := GetDllProc(FHandle, 'H5Tset_sign');
  @FH5Tset_fields := GetDllProc(FHandle, 'H5Tset_fields');
  @FH5Tset_ebias := GetDllProc(FHandle, 'H5Tset_ebias');
  @FH5Tset_norm := GetDllProc(FHandle, 'H5Tset_norm');
  @FH5Tset_inpad := GetDllProc(FHandle, 'H5Tset_inpad');
  @FH5Tset_cset := GetDllProc(FHandle, 'H5Tset_cset');
  @FH5Tset_strpad := GetDllProc(FHandle, 'H5Tset_strpad');
  @FH5Tregister := GetDllProc(FHandle, 'H5Tregister');
  @FH5Tunregister := GetDllProc(FHandle, 'H5Tunregister');
  @FH5Tfind := GetDllProc(FHandle, 'H5Tfind');
  @FH5Tcompiler_conv := GetDllProc(FHandle, 'H5Tcompiler_conv');
  @FH5Tconvert := GetDllProc(FHandle, 'H5Tconvert');
  @FH5Dcreate2 := GetDllProc(FHandle, 'H5Dcreate2');
  @FH5Dcreate_anon := GetDllProc(FHandle, 'H5Dcreate_anon');
  @FH5Dopen2 := GetDllProc(FHandle, 'H5Dopen2');
  @FH5Dclose := GetDllProc(FHandle, 'H5Dclose');
  @FH5Dget_space := GetDllProc(FHandle, 'H5Dget_space');
  @FH5Dget_space_status := GetDllProc(FHandle, 'H5Dget_space_status');
  @FH5Dget_type := GetDllProc(FHandle, 'H5Dget_type');
  @FH5Dget_create_plist := GetDllProc(FHandle, 'H5Dget_create_plist');
  @FH5Dget_access_plist := GetDllProc(FHandle, 'H5Dget_access_plist');
  @FH5Dget_storage_size := GetDllProc(FHandle, 'H5Dget_storage_size');
  @FH5Dget_offset := GetDllProc(FHandle, 'H5Dget_offset');
  @FH5Dread := GetDllProc(FHandle, 'H5Dread');
  @FH5Dwrite := GetDllProc(FHandle, 'H5Dwrite');
  @FH5Diterate := GetDllProc(FHandle, 'H5Diterate');
  @FH5Dvlen_reclaim := GetDllProc(FHandle, 'H5Dvlen_reclaim');
  @FH5Dvlen_get_buf_size := GetDllProc(FHandle, 'H5Dvlen_get_buf_size');
  @FH5Dfill := GetDllProc(FHandle, 'H5Dfill');
  @FH5Dset_extent := GetDllProc(FHandle, 'H5Dset_extent');
  @FH5Dflush := GetDllProc(FHandle, 'H5Dflush');
  @FH5Drefresh := GetDllProc(FHandle, 'H5Drefresh');
  @FH5Dscatter := GetDllProc(FHandle, 'H5Dscatter');
  @FH5Dgather := GetDllProc(FHandle, 'H5Dgather');
  @FH5Ddebug := GetDllProc(FHandle, 'H5Ddebug');
  @FH5Dformat_convert := GetDllProc(FHandle, 'H5Dformat_convert');
  @FH5Dget_chunk_index_type := GetDllProc(FHandle, 'H5Dget_chunk_index_type');
  @FH5Eregister_class := GetDllProc(FHandle, 'H5Eregister_class');
  @FH5Eunregister_class := GetDllProc(FHandle, 'H5Eunregister_class');
  @FH5Eclose_msg := GetDllProc(FHandle, 'H5Eclose_msg');
  @FH5Ecreate_msg := GetDllProc(FHandle, 'H5Ecreate_msg');
  @FH5Ecreate_stack := GetDllProc(FHandle, 'H5Ecreate_stack');
  @FH5Eget_current_stack := GetDllProc(FHandle, 'H5Eget_current_stack');
  @FH5Eclose_stack := GetDllProc(FHandle, 'H5Eclose_stack');
  @FH5Eget_class_name := GetDllProc(FHandle, 'H5Eget_class_name');
  @FH5Eset_current_stack := GetDllProc(FHandle, 'H5Eset_current_stack');
  @FH5Epop := GetDllProc(FHandle, 'H5Epop');
  @FH5Eprint2 := GetDllProc(FHandle, 'H5Eprint2');
  @FH5Ewalk2 := GetDllProc(FHandle, 'H5Ewalk2');
  @FH5Eget_auto2 := GetDllProc(FHandle, 'H5Eget_auto2');
  @FH5Eset_auto2 := GetDllProc(FHandle, 'H5Eset_auto2');
  @FH5Eclear2 := GetDllProc(FHandle, 'H5Eclear2');
  @FH5Eauto_is_v2 := GetDllProc(FHandle, 'H5Eauto_is_v2');
  @FH5Eget_msg := GetDllProc(FHandle, 'H5Eget_msg');
  @FH5Eget_num := GetDllProc(FHandle, 'H5Eget_num');
  @FH5Screate := GetDllProc(FHandle, 'H5Screate');
  @FH5Screate_simple := GetDllProc(FHandle, 'H5Screate_simple');
  @FH5Sset_extent_simple := GetDllProc(FHandle, 'H5Sset_extent_simple');
  @FH5Scopy := GetDllProc(FHandle, 'H5Scopy');
  @FH5Sclose := GetDllProc(FHandle, 'H5Sclose');
  @FH5Sencode := GetDllProc(FHandle, 'H5Sencode');
  @FH5Sdecode := GetDllProc(FHandle, 'H5Sdecode');
  @FH5Sget_simple_extent_npoints := GetDllProc(FHandle, 'H5Sget_simple_extent_npoints');
  @FH5Sget_simple_extent_ndims := GetDllProc(FHandle, 'H5Sget_simple_extent_ndims');
  @FH5Sget_simple_extent_dims := GetDllProc(FHandle, 'H5Sget_simple_extent_dims');
  @FH5Sis_simple := GetDllProc(FHandle, 'H5Sis_simple');
  @FH5Sget_select_npoints := GetDllProc(FHandle, 'H5Sget_select_npoints');
  @FH5Sselect_hyperslab := GetDllProc(FHandle, 'H5Sselect_hyperslab');
  @FH5Sselect_elements := GetDllProc(FHandle, 'H5Sselect_elements');
  @FH5Sget_simple_extent_type := GetDllProc(FHandle, 'H5Sget_simple_extent_type');
  @FH5Sset_extent_none := GetDllProc(FHandle, 'H5Sset_extent_none');
  @FH5Sextent_copy := GetDllProc(FHandle, 'H5Sextent_copy');
  @FH5Sextent_equal := GetDllProc(FHandle, 'H5Sextent_equal');
  @FH5Sselect_all := GetDllProc(FHandle, 'H5Sselect_all');
  @FH5Sselect_none := GetDllProc(FHandle, 'H5Sselect_none');
  @FH5Soffset_simple := GetDllProc(FHandle, 'H5Soffset_simple');
  @FH5Sselect_valid := GetDllProc(FHandle, 'H5Sselect_valid');
  @FH5Sis_regular_hyperslab := GetDllProc(FHandle, 'H5Sis_regular_hyperslab');
  @FH5Sget_regular_hyperslab := GetDllProc(FHandle, 'H5Sget_regular_hyperslab');
  @FH5Sget_select_hyper_nblocks := GetDllProc(FHandle, 'H5Sget_select_hyper_nblocks');
  @FH5Sget_select_elem_npoints := GetDllProc(FHandle, 'H5Sget_select_elem_npoints');
  @FH5Sget_select_hyper_blocklist := GetDllProc(FHandle, 'H5Sget_select_hyper_blocklist');
  @FH5Sget_select_elem_pointlist := GetDllProc(FHandle, 'H5Sget_select_elem_pointlist');
  @FH5Sget_select_bounds := GetDllProc(FHandle, 'H5Sget_select_bounds');
  @FH5Sget_select_type := GetDllProc(FHandle, 'H5Sget_select_type');
  @FH5Lmove := GetDllProc(FHandle, 'H5Lmove');
  @FH5Lcopy := GetDllProc(FHandle, 'H5Lcopy');
  @FH5Lcreate_hard := GetDllProc(FHandle, 'H5Lcreate_hard');
  @FH5Lcreate_soft := GetDllProc(FHandle, 'H5Lcreate_soft');
  @FH5Ldelete := GetDllProc(FHandle, 'H5Ldelete');
  @FH5Ldelete_by_idx := GetDllProc(FHandle, 'H5Ldelete_by_idx');
  @FH5Lget_val := GetDllProc(FHandle, 'H5Lget_val');
  @FH5Lget_val_by_idx := GetDllProc(FHandle, 'H5Lget_val_by_idx');
  @FH5Lexists := GetDllProc(FHandle, 'H5Lexists');
  @FH5Lget_info := GetDllProc(FHandle, 'H5Lget_info');
  @FH5Lget_info_by_idx := GetDllProc(FHandle, 'H5Lget_info_by_idx');
  @FH5Lget_name_by_idx := GetDllProc(FHandle, 'H5Lget_name_by_idx');
  @FH5Literate := GetDllProc(FHandle, 'H5Literate');
  @FH5Literate_by_name := GetDllProc(FHandle, 'H5Literate_by_name');
  @FH5Lvisit := GetDllProc(FHandle, 'H5Lvisit');
  @FH5Lvisit_by_name := GetDllProc(FHandle, 'H5Lvisit_by_name');
  @FH5Lcreate_ud := GetDllProc(FHandle, 'H5Lcreate_ud');
  @FH5Lregister := GetDllProc(FHandle, 'H5Lregister');
  @FH5Lunregister := GetDllProc(FHandle, 'H5Lunregister');
  @FH5Lis_registered := GetDllProc(FHandle, 'H5Lis_registered');
  @FH5Lunpack_elink_val := GetDllProc(FHandle, 'H5Lunpack_elink_val');
  @FH5Lcreate_external := GetDllProc(FHandle, 'H5Lcreate_external');
  @FH5Oopen := GetDllProc(FHandle, 'H5Oopen');
  @FH5Oopen_by_addr := GetDllProc(FHandle, 'H5Oopen_by_addr');
  @FH5Oopen_by_idx := GetDllProc(FHandle, 'H5Oopen_by_idx');
  @FH5Oexists_by_name := GetDllProc(FHandle, 'H5Oexists_by_name');
  @FH5Oget_info := GetDllProc(FHandle, 'H5Oget_info');
  @FH5Oget_info_by_name := GetDllProc(FHandle, 'H5Oget_info_by_name');
  @FH5Oget_info_by_idx := GetDllProc(FHandle, 'H5Oget_info_by_idx');
  @FH5Olink := GetDllProc(FHandle, 'H5Olink');
  @FH5Oincr_refcount := GetDllProc(FHandle, 'H5Oincr_refcount');
  @FH5Odecr_refcount := GetDllProc(FHandle, 'H5Odecr_refcount');
  @FH5Ocopy := GetDllProc(FHandle, 'H5Ocopy');
  @FH5Oset_comment := GetDllProc(FHandle, 'H5Oset_comment');
  @FH5Oset_comment_by_name := GetDllProc(FHandle, 'H5Oset_comment_by_name');
  @FH5Oget_comment := GetDllProc(FHandle, 'H5Oget_comment');
  @FH5Oget_comment_by_name := GetDllProc(FHandle, 'H5Oget_comment_by_name');
  @FH5Ovisit := GetDllProc(FHandle, 'H5Ovisit');
  @FH5Ovisit_by_name := GetDllProc(FHandle, 'H5Ovisit_by_name');
  @FH5Oclose := GetDllProc(FHandle, 'H5Oclose');
  @FH5Oflush := GetDllProc(FHandle, 'H5Oflush');
  @FH5Orefresh := GetDllProc(FHandle, 'H5Orefresh');
  @FH5Odisable_mdc_flushes := GetDllProc(FHandle, 'H5Odisable_mdc_flushes');
  @FH5Oenable_mdc_flushes := GetDllProc(FHandle, 'H5Oenable_mdc_flushes');
  @FH5Oare_mdc_flushes_disabled := GetDllProc(FHandle, 'H5Oare_mdc_flushes_disabled');
  @FH5Fis_hdf5 := GetDllProc(FHandle, 'H5Fis_hdf5');
  @FH5Fcreate := GetDllProc(FHandle, 'H5Fcreate');
  @FH5Fopen := GetDllProc(FHandle, 'H5Fopen');
  @FH5Freopen := GetDllProc(FHandle, 'H5Freopen');
  @FH5Fflush := GetDllProc(FHandle, 'H5Fflush');
  @FH5Fclose := GetDllProc(FHandle, 'H5Fclose');
  @FH5Fget_create_plist := GetDllProc(FHandle, 'H5Fget_create_plist');
  @FH5Fget_access_plist := GetDllProc(FHandle, 'H5Fget_access_plist');
  @FH5Fget_intent := GetDllProc(FHandle, 'H5Fget_intent');
  @FH5Fget_obj_count := GetDllProc(FHandle, 'H5Fget_obj_count');
  @FH5Fget_obj_ids := GetDllProc(FHandle, 'H5Fget_obj_ids');
  @FH5Fget_vfd_handle := GetDllProc(FHandle, 'H5Fget_vfd_handle');
  @FH5Fmount := GetDllProc(FHandle, 'H5Fmount');
  @FH5Funmount := GetDllProc(FHandle, 'H5Funmount');
  @FH5Fget_freespace := GetDllProc(FHandle, 'H5Fget_freespace');
  @FH5Fget_filesize := GetDllProc(FHandle, 'H5Fget_filesize');
  @FH5Fget_file_image := GetDllProc(FHandle, 'H5Fget_file_image');
  @FH5Fget_mdc_config := GetDllProc(FHandle, 'H5Fget_mdc_config');
  @FH5Fset_mdc_config := GetDllProc(FHandle, 'H5Fset_mdc_config');
  @FH5Fget_mdc_hit_rate := GetDllProc(FHandle, 'H5Fget_mdc_hit_rate');
  @FH5Fget_mdc_size := GetDllProc(FHandle, 'H5Fget_mdc_size');
  @FH5Freset_mdc_hit_rate_stats := GetDllProc(FHandle, 'H5Freset_mdc_hit_rate_stats');
  @FH5Fget_name := GetDllProc(FHandle, 'H5Fget_name');
  @FH5Fget_info2 := GetDllProc(FHandle, 'H5Fget_info2');
  @FH5Fget_metadata_read_retry_info := GetDllProc(FHandle, 'H5Fget_metadata_read_retry_info');
  @FH5Fstart_swmr_write := GetDllProc(FHandle, 'H5Fstart_swmr_write');
  @FH5Fget_free_sections := GetDllProc(FHandle, 'H5Fget_free_sections');
  @FH5Fclear_elink_file_cache := GetDllProc(FHandle, 'H5Fclear_elink_file_cache');
  @FH5Fstart_mdc_logging := GetDllProc(FHandle, 'H5Fstart_mdc_logging');
  @FH5Fstop_mdc_logging := GetDllProc(FHandle, 'H5Fstop_mdc_logging');
  @FH5Fget_mdc_logging_status := GetDllProc(FHandle, 'H5Fget_mdc_logging_status');
  @FH5Fformat_convert := GetDllProc(FHandle, 'H5Fformat_convert');
  @FH5Acreate2 := GetDllProc(FHandle, 'H5Acreate2');
  @FH5Acreate_by_name := GetDllProc(FHandle, 'H5Acreate_by_name');
  @FH5Aopen := GetDllProc(FHandle, 'H5Aopen');
  @FH5Aopen_by_name := GetDllProc(FHandle, 'H5Aopen_by_name');
  @FH5Aopen_by_idx := GetDllProc(FHandle, 'H5Aopen_by_idx');
  @FH5Awrite := GetDllProc(FHandle, 'H5Awrite');
  @FH5Aread := GetDllProc(FHandle, 'H5Aread');
  @FH5Aclose := GetDllProc(FHandle, 'H5Aclose');
  @FH5Aget_space := GetDllProc(FHandle, 'H5Aget_space');
  @FH5Aget_type := GetDllProc(FHandle, 'H5Aget_type');
  @FH5Aget_create_plist := GetDllProc(FHandle, 'H5Aget_create_plist');
  @FH5Aget_name := GetDllProc(FHandle, 'H5Aget_name');
  @FH5Aget_name_by_idx := GetDllProc(FHandle, 'H5Aget_name_by_idx');
  @FH5Aget_storage_size := GetDllProc(FHandle, 'H5Aget_storage_size');
  @FH5Aget_info := GetDllProc(FHandle, 'H5Aget_info');
  @FH5Aget_info_by_name := GetDllProc(FHandle, 'H5Aget_info_by_name');
  @FH5Aget_info_by_idx := GetDllProc(FHandle, 'H5Aget_info_by_idx');
  @FH5Arename := GetDllProc(FHandle, 'H5Arename');
  @FH5Arename_by_name := GetDllProc(FHandle, 'H5Arename_by_name');
  @FH5Aiterate2 := GetDllProc(FHandle, 'H5Aiterate2');
  @FH5Aiterate_by_name := GetDllProc(FHandle, 'H5Aiterate_by_name');
  @FH5Adelete := GetDllProc(FHandle, 'H5Adelete');
  @FH5Adelete_by_name := GetDllProc(FHandle, 'H5Adelete_by_name');
  @FH5Adelete_by_idx := GetDllProc(FHandle, 'H5Adelete_by_idx');
  @FH5Aexists := GetDllProc(FHandle, 'H5Aexists');
  @FH5Aexists_by_name := GetDllProc(FHandle, 'H5Aexists_by_name');
  @FH5FDregister := GetDllProc(FHandle, 'H5FDregister');
  @FH5FDunregister := GetDllProc(FHandle, 'H5FDunregister');
  @FH5FDopen := GetDllProc(FHandle, 'H5FDopen');
  @FH5FDclose := GetDllProc(FHandle, 'H5FDclose');
  @FH5FDcmp := GetDllProc(FHandle, 'H5FDcmp');
  @FH5FDquery := GetDllProc(FHandle, 'H5FDquery');
  @FH5FDalloc := GetDllProc(FHandle, 'H5FDalloc');
  @FH5FDfree := GetDllProc(FHandle, 'H5FDfree');
  @FH5FDget_eoa := GetDllProc(FHandle, 'H5FDget_eoa');
  @FH5FDset_eoa := GetDllProc(FHandle, 'H5FDset_eoa');
  @FH5FDget_eof := GetDllProc(FHandle, 'H5FDget_eof');
  @FH5FDget_vfd_handle := GetDllProc(FHandle, 'H5FDget_vfd_handle');
  @FH5FDread := GetDllProc(FHandle, 'H5FDread');
  @FH5FDwrite := GetDllProc(FHandle, 'H5FDwrite');
  @FH5FDflush := GetDllProc(FHandle, 'H5FDflush');
  @FH5FDtruncate := GetDllProc(FHandle, 'H5FDtruncate');
  @FH5FDlock := GetDllProc(FHandle, 'H5FDlock');
  @FH5FDunlock := GetDllProc(FHandle, 'H5FDunlock');
  @FH5Gcreate2 := GetDllProc(FHandle, 'H5Gcreate2');
  @FH5Gcreate_anon := GetDllProc(FHandle, 'H5Gcreate_anon');
  @FH5Gopen2 := GetDllProc(FHandle, 'H5Gopen2');
  @FH5Gget_create_plist := GetDllProc(FHandle, 'H5Gget_create_plist');
  @FH5Gget_info := GetDllProc(FHandle, 'H5Gget_info');
  @FH5Gget_info_by_name := GetDllProc(FHandle, 'H5Gget_info_by_name');
  @FH5Gget_info_by_idx := GetDllProc(FHandle, 'H5Gget_info_by_idx');
  @FH5Gclose := GetDllProc(FHandle, 'H5Gclose');
  @FH5Gflush := GetDllProc(FHandle, 'H5Gflush');
  @FH5Grefresh := GetDllProc(FHandle, 'H5Grefresh');
  @FH5Rcreate := GetDllProc(FHandle, 'H5Rcreate');
  @FH5Rdereference2 := GetDllProc(FHandle, 'H5Rdereference2');
  @FH5Rget_region := GetDllProc(FHandle, 'H5Rget_region');
  @FH5Rget_obj_type2 := GetDllProc(FHandle, 'H5Rget_obj_type2');
  @FH5Rget_name := GetDllProc(FHandle, 'H5Rget_name');
  @FH5Pcreate_class := GetDllProc(FHandle, 'H5Pcreate_class');
  @FH5Pget_class_name := GetDllProc(FHandle, 'H5Pget_class_name');
  @FH5Pcreate := GetDllProc(FHandle, 'H5Pcreate');
  @FH5Pregister2 := GetDllProc(FHandle, 'H5Pregister2');
  @FH5Pinsert2 := GetDllProc(FHandle, 'H5Pinsert2');
  @FH5Pset := GetDllProc(FHandle, 'H5Pset');
  @FH5Pexist := GetDllProc(FHandle, 'H5Pexist');
  @FH5Pencode := GetDllProc(FHandle, 'H5Pencode');
  @FH5Pdecode := GetDllProc(FHandle, 'H5Pdecode');
  @FH5Pget_size := GetDllProc(FHandle, 'H5Pget_size');
  @FH5Pget_nprops := GetDllProc(FHandle, 'H5Pget_nprops');
  @FH5Pget_class := GetDllProc(FHandle, 'H5Pget_class');
  @FH5Pget_class_parent := GetDllProc(FHandle, 'H5Pget_class_parent');
  @FH5Pget := GetDllProc(FHandle, 'H5Pget');
  @FH5Pequal := GetDllProc(FHandle, 'H5Pequal');
  @FH5Pisa_class := GetDllProc(FHandle, 'H5Pisa_class');
  @FH5Piterate := GetDllProc(FHandle, 'H5Piterate');
  @FH5Pcopy_prop := GetDllProc(FHandle, 'H5Pcopy_prop');
  @FH5Premove := GetDllProc(FHandle, 'H5Premove');
  @FH5Punregister := GetDllProc(FHandle, 'H5Punregister');
  @FH5Pclose_class := GetDllProc(FHandle, 'H5Pclose_class');
  @FH5Pclose := GetDllProc(FHandle, 'H5Pclose');
  @FH5Pcopy := GetDllProc(FHandle, 'H5Pcopy');
  @FH5Pset_attr_phase_change := GetDllProc(FHandle, 'H5Pset_attr_phase_change');
  @FH5Pget_attr_phase_change := GetDllProc(FHandle, 'H5Pget_attr_phase_change');
  @FH5Pset_attr_creation_order := GetDllProc(FHandle, 'H5Pset_attr_creation_order');
  @FH5Pget_attr_creation_order := GetDllProc(FHandle, 'H5Pget_attr_creation_order');
  @FH5Pset_obj_track_times := GetDllProc(FHandle, 'H5Pset_obj_track_times');
  @FH5Pget_obj_track_times := GetDllProc(FHandle, 'H5Pget_obj_track_times');
  @FH5Pmodify_filter := GetDllProc(FHandle, 'H5Pmodify_filter');
  @FH5Pset_filter := GetDllProc(FHandle, 'H5Pset_filter');
  @FH5Pget_nfilters := GetDllProc(FHandle, 'H5Pget_nfilters');
  @FH5Pget_filter2 := GetDllProc(FHandle, 'H5Pget_filter2');
  @FH5Pget_filter_by_id2 := GetDllProc(FHandle, 'H5Pget_filter_by_id2');
  @FH5Pall_filters_avail := GetDllProc(FHandle, 'H5Pall_filters_avail');
  @FH5Premove_filter := GetDllProc(FHandle, 'H5Premove_filter');
  @FH5Pset_deflate := GetDllProc(FHandle, 'H5Pset_deflate');
  @FH5Pset_fletcher32 := GetDllProc(FHandle, 'H5Pset_fletcher32');
  @FH5Pset_userblock := GetDllProc(FHandle, 'H5Pset_userblock');
  @FH5Pget_userblock := GetDllProc(FHandle, 'H5Pget_userblock');
  @FH5Pset_sizes := GetDllProc(FHandle, 'H5Pset_sizes');
  @FH5Pget_sizes := GetDllProc(FHandle, 'H5Pget_sizes');
  @FH5Pset_sym_k := GetDllProc(FHandle, 'H5Pset_sym_k');
  @FH5Pget_sym_k := GetDllProc(FHandle, 'H5Pget_sym_k');
  @FH5Pset_istore_k := GetDllProc(FHandle, 'H5Pset_istore_k');
  @FH5Pget_istore_k := GetDllProc(FHandle, 'H5Pget_istore_k');
  @FH5Pset_shared_mesg_nindexes := GetDllProc(FHandle, 'H5Pset_shared_mesg_nindexes');
  @FH5Pget_shared_mesg_nindexes := GetDllProc(FHandle, 'H5Pget_shared_mesg_nindexes');
  @FH5Pset_shared_mesg_index := GetDllProc(FHandle, 'H5Pset_shared_mesg_index');
  @FH5Pget_shared_mesg_index := GetDllProc(FHandle, 'H5Pget_shared_mesg_index');
  @FH5Pset_shared_mesg_phase_change := GetDllProc(FHandle, 'H5Pset_shared_mesg_phase_change');
  @FH5Pget_shared_mesg_phase_change := GetDllProc(FHandle, 'H5Pget_shared_mesg_phase_change');
  @FH5Pset_file_space := GetDllProc(FHandle, 'H5Pset_file_space');
  @FH5Pget_file_space := GetDllProc(FHandle, 'H5Pget_file_space');
  @FH5Pset_alignment := GetDllProc(FHandle, 'H5Pset_alignment');
  @FH5Pget_alignment := GetDllProc(FHandle, 'H5Pget_alignment');
  @FH5Pset_driver := GetDllProc(FHandle, 'H5Pset_driver');
  @FH5Pget_driver := GetDllProc(FHandle, 'H5Pget_driver');
  @FH5Pget_driver_info := GetDllProc(FHandle, 'H5Pget_driver_info');
  @FH5Pset_family_offset := GetDllProc(FHandle, 'H5Pset_family_offset');
  @FH5Pget_family_offset := GetDllProc(FHandle, 'H5Pget_family_offset');
  @FH5Pset_multi_type := GetDllProc(FHandle, 'H5Pset_multi_type');
  @FH5Pget_multi_type := GetDllProc(FHandle, 'H5Pget_multi_type');
  @FH5Pset_cache := GetDllProc(FHandle, 'H5Pset_cache');
  @FH5Pget_cache := GetDllProc(FHandle, 'H5Pget_cache');
  @FH5Pset_mdc_config := GetDllProc(FHandle, 'H5Pset_mdc_config');
  @FH5Pget_mdc_config := GetDllProc(FHandle, 'H5Pget_mdc_config');
  @FH5Pset_gc_references := GetDllProc(FHandle, 'H5Pset_gc_references');
  @FH5Pget_gc_references := GetDllProc(FHandle, 'H5Pget_gc_references');
  @FH5Pset_fclose_degree := GetDllProc(FHandle, 'H5Pset_fclose_degree');
  @FH5Pget_fclose_degree := GetDllProc(FHandle, 'H5Pget_fclose_degree');
  @FH5Pset_meta_block_size := GetDllProc(FHandle, 'H5Pset_meta_block_size');
  @FH5Pget_meta_block_size := GetDllProc(FHandle, 'H5Pget_meta_block_size');
  @FH5Pset_sieve_buf_size := GetDllProc(FHandle, 'H5Pset_sieve_buf_size');
  @FH5Pget_sieve_buf_size := GetDllProc(FHandle, 'H5Pget_sieve_buf_size');
  @FH5Pset_small_data_block_size := GetDllProc(FHandle, 'H5Pset_small_data_block_size');
  @FH5Pget_small_data_block_size := GetDllProc(FHandle, 'H5Pget_small_data_block_size');
  @FH5Pset_libver_bounds := GetDllProc(FHandle, 'H5Pset_libver_bounds');
  @FH5Pget_libver_bounds := GetDllProc(FHandle, 'H5Pget_libver_bounds');
  @FH5Pset_elink_file_cache_size := GetDllProc(FHandle, 'H5Pset_elink_file_cache_size');
  @FH5Pget_elink_file_cache_size := GetDllProc(FHandle, 'H5Pget_elink_file_cache_size');
  @FH5Pset_file_image := GetDllProc(FHandle, 'H5Pset_file_image');
  @FH5Pget_file_image := GetDllProc(FHandle, 'H5Pget_file_image');
  @FH5Pset_file_image_callbacks := GetDllProc(FHandle, 'H5Pset_file_image_callbacks');
  @FH5Pget_file_image_callbacks := GetDllProc(FHandle, 'H5Pget_file_image_callbacks');
  @FH5Pset_core_write_tracking := GetDllProc(FHandle, 'H5Pset_core_write_tracking');
  @FH5Pget_core_write_tracking := GetDllProc(FHandle, 'H5Pget_core_write_tracking');
  @FH5Pset_metadata_read_attempts := GetDllProc(FHandle, 'H5Pset_metadata_read_attempts');
  @FH5Pget_metadata_read_attempts := GetDllProc(FHandle, 'H5Pget_metadata_read_attempts');
  @FH5Pset_object_flush_cb := GetDllProc(FHandle, 'H5Pset_object_flush_cb');
  @FH5Pget_object_flush_cb := GetDllProc(FHandle, 'H5Pget_object_flush_cb');
  @FH5Pset_mdc_log_options := GetDllProc(FHandle, 'H5Pset_mdc_log_options');
  @FH5Pget_mdc_log_options := GetDllProc(FHandle, 'H5Pget_mdc_log_options');
  @FH5Pset_layout := GetDllProc(FHandle, 'H5Pset_layout');
  @FH5Pget_layout := GetDllProc(FHandle, 'H5Pget_layout');
  @FH5Pset_chunk := GetDllProc(FHandle, 'H5Pset_chunk');
  @FH5Pget_chunk := GetDllProc(FHandle, 'H5Pget_chunk');
  @FH5Pset_virtual := GetDllProc(FHandle, 'H5Pset_virtual');
  @FH5Pget_virtual_count := GetDllProc(FHandle, 'H5Pget_virtual_count');
  @FH5Pget_virtual_vspace := GetDllProc(FHandle, 'H5Pget_virtual_vspace');
  @FH5Pget_virtual_srcspace := GetDllProc(FHandle, 'H5Pget_virtual_srcspace');
  @FH5Pget_virtual_filename := GetDllProc(FHandle, 'H5Pget_virtual_filename');
  @FH5Pget_virtual_dsetname := GetDllProc(FHandle, 'H5Pget_virtual_dsetname');
  @FH5Pset_external := GetDllProc(FHandle, 'H5Pset_external');
  @FH5Pset_chunk_opts := GetDllProc(FHandle, 'H5Pset_chunk_opts');
  @FH5Pget_chunk_opts := GetDllProc(FHandle, 'H5Pget_chunk_opts');
  @FH5Pget_external_count := GetDllProc(FHandle, 'H5Pget_external_count');
  @FH5Pget_external := GetDllProc(FHandle, 'H5Pget_external');
  @FH5Pset_szip := GetDllProc(FHandle, 'H5Pset_szip');
  @FH5Pset_shuffle := GetDllProc(FHandle, 'H5Pset_shuffle');
  @FH5Pset_nbit := GetDllProc(FHandle, 'H5Pset_nbit');
  @FH5Pset_scaleoffset := GetDllProc(FHandle, 'H5Pset_scaleoffset');
  @FH5Pset_fill_value := GetDllProc(FHandle, 'H5Pset_fill_value');
  @FH5Pget_fill_value := GetDllProc(FHandle, 'H5Pget_fill_value');
  @FH5Pfill_value_defined := GetDllProc(FHandle, 'H5Pfill_value_defined');
  @FH5Pset_alloc_time := GetDllProc(FHandle, 'H5Pset_alloc_time');
  @FH5Pget_alloc_time := GetDllProc(FHandle, 'H5Pget_alloc_time');
  @FH5Pset_fill_time := GetDllProc(FHandle, 'H5Pset_fill_time');
  @FH5Pget_fill_time := GetDllProc(FHandle, 'H5Pget_fill_time');
  @FH5Pset_chunk_cache := GetDllProc(FHandle, 'H5Pset_chunk_cache');
  @FH5Pget_chunk_cache := GetDllProc(FHandle, 'H5Pget_chunk_cache');
  @FH5Pset_virtual_view := GetDllProc(FHandle, 'H5Pset_virtual_view');
  @FH5Pget_virtual_view := GetDllProc(FHandle, 'H5Pget_virtual_view');
  @FH5Pset_virtual_printf_gap := GetDllProc(FHandle, 'H5Pset_virtual_printf_gap');
  @FH5Pget_virtual_printf_gap := GetDllProc(FHandle, 'H5Pget_virtual_printf_gap');
  @FH5Pset_append_flush := GetDllProc(FHandle, 'H5Pset_append_flush');
  @FH5Pget_append_flush := GetDllProc(FHandle, 'H5Pget_append_flush');
  @FH5Pset_efile_prefix := GetDllProc(FHandle, 'H5Pset_efile_prefix');
  @FH5Pget_efile_prefix := GetDllProc(FHandle, 'H5Pget_efile_prefix');
  @FH5Pset_data_transform := GetDllProc(FHandle, 'H5Pset_data_transform');
  @FH5Pget_data_transform := GetDllProc(FHandle, 'H5Pget_data_transform');
  @FH5Pset_buffer := GetDllProc(FHandle, 'H5Pset_buffer');
  @FH5Pget_buffer := GetDllProc(FHandle, 'H5Pget_buffer');
  @FH5Pset_preserve := GetDllProc(FHandle, 'H5Pset_preserve');
  @FH5Pget_preserve := GetDllProc(FHandle, 'H5Pget_preserve');
  @FH5Pset_edc_check := GetDllProc(FHandle, 'H5Pset_edc_check');
  @FH5Pget_edc_check := GetDllProc(FHandle, 'H5Pget_edc_check');
  @FH5Pset_filter_callback := GetDllProc(FHandle, 'H5Pset_filter_callback');
  @FH5Pset_btree_ratios := GetDllProc(FHandle, 'H5Pset_btree_ratios');
  @FH5Pget_btree_ratios := GetDllProc(FHandle, 'H5Pget_btree_ratios');
  @FH5Pset_vlen_mem_manager := GetDllProc(FHandle, 'H5Pset_vlen_mem_manager');
  @FH5Pget_vlen_mem_manager := GetDllProc(FHandle, 'H5Pget_vlen_mem_manager');
  @FH5Pset_hyper_vector_size := GetDllProc(FHandle, 'H5Pset_hyper_vector_size');
  @FH5Pget_hyper_vector_size := GetDllProc(FHandle, 'H5Pget_hyper_vector_size');
  @FH5Pset_type_conv_cb := GetDllProc(FHandle, 'H5Pset_type_conv_cb');
  @FH5Pget_type_conv_cb := GetDllProc(FHandle, 'H5Pget_type_conv_cb');
  @FH5Pset_create_intermediate_group := GetDllProc(FHandle, 'H5Pset_create_intermediate_group');
  @FH5Pget_create_intermediate_group := GetDllProc(FHandle, 'H5Pget_create_intermediate_group');
  @FH5Pset_local_heap_size_hint := GetDllProc(FHandle, 'H5Pset_local_heap_size_hint');
  @FH5Pget_local_heap_size_hint := GetDllProc(FHandle, 'H5Pget_local_heap_size_hint');
  @FH5Pset_link_phase_change := GetDllProc(FHandle, 'H5Pset_link_phase_change');
  @FH5Pget_link_phase_change := GetDllProc(FHandle, 'H5Pget_link_phase_change');
  @FH5Pset_est_link_info := GetDllProc(FHandle, 'H5Pset_est_link_info');
  @FH5Pget_est_link_info := GetDllProc(FHandle, 'H5Pget_est_link_info');
  @FH5Pset_link_creation_order := GetDllProc(FHandle, 'H5Pset_link_creation_order');
  @FH5Pget_link_creation_order := GetDllProc(FHandle, 'H5Pget_link_creation_order');
  @FH5Pset_char_encoding := GetDllProc(FHandle, 'H5Pset_char_encoding');
  @FH5Pget_char_encoding := GetDllProc(FHandle, 'H5Pget_char_encoding');
  @FH5Pset_nlinks := GetDllProc(FHandle, 'H5Pset_nlinks');
  @FH5Pget_nlinks := GetDllProc(FHandle, 'H5Pget_nlinks');
  @FH5Pset_elink_prefix := GetDllProc(FHandle, 'H5Pset_elink_prefix');
  @FH5Pget_elink_prefix := GetDllProc(FHandle, 'H5Pget_elink_prefix');
  @FH5Pget_elink_fapl := GetDllProc(FHandle, 'H5Pget_elink_fapl');
  @FH5Pset_elink_fapl := GetDllProc(FHandle, 'H5Pset_elink_fapl');
  @FH5Pset_elink_acc_flags := GetDllProc(FHandle, 'H5Pset_elink_acc_flags');
  @FH5Pget_elink_acc_flags := GetDllProc(FHandle, 'H5Pget_elink_acc_flags');
  @FH5Pset_elink_cb := GetDllProc(FHandle, 'H5Pset_elink_cb');
  @FH5Pget_elink_cb := GetDllProc(FHandle, 'H5Pget_elink_cb');
  @FH5Pset_copy_object := GetDllProc(FHandle, 'H5Pset_copy_object');
  @FH5Pget_copy_object := GetDllProc(FHandle, 'H5Pget_copy_object');
  @FH5Padd_merge_committed_dtype_path := GetDllProc(FHandle, 'H5Padd_merge_committed_dtype_path');
  @FH5Pfree_merge_committed_dtype_paths := GetDllProc(FHandle, 'H5Pfree_merge_committed_dtype_paths');
  @FH5Pset_mcdt_search_cb := GetDllProc(FHandle, 'H5Pset_mcdt_search_cb');
  @FH5Pget_mcdt_search_cb := GetDllProc(FHandle, 'H5Pget_mcdt_search_cb');

  H5open;
  FH5T_IEEE_F32BE := Phid_t(GetDllProc(FHandle, 'H5T_IEEE_F32BE_g'))^;
  FH5T_IEEE_F32LE := Phid_t(GetDllProc(FHandle, 'H5T_IEEE_F32LE_g'))^;
  FH5T_IEEE_F64BE := Phid_t(GetDllProc(FHandle, 'H5T_IEEE_F64BE_g'))^;
  FH5T_IEEE_F64LE := Phid_t(GetDllProc(FHandle, 'H5T_IEEE_F64LE_g'))^;
  FH5T_STD_I8BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I8BE_g'))^;
  FH5T_STD_I8LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I8LE_g'))^;
  FH5T_STD_I16BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I16BE_g'))^;
  FH5T_STD_I16LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I16LE_g'))^;
  FH5T_STD_I32BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I32BE_g'))^;
  FH5T_STD_I32LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I32LE_g'))^;
  FH5T_STD_I64BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I64BE_g'))^;
  FH5T_STD_I64LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_I64LE_g'))^;
  FH5T_STD_U8BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U8BE_g'))^;
  FH5T_STD_U8LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U8LE_g'))^;
  FH5T_STD_U16BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U16BE_g'))^;
  FH5T_STD_U16LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U16LE_g'))^;
  FH5T_STD_U32BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U32BE_g'))^;
  FH5T_STD_U32LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U32LE_g'))^;
  FH5T_STD_U64BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U64BE_g'))^;
  FH5T_STD_U64LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_U64LE_g'))^;
  FH5T_STD_B8BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B8BE_g'))^;
  FH5T_STD_B8LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B8LE_g'))^;
  FH5T_STD_B16BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B16BE_g'))^;
  FH5T_STD_B16LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B16LE_g'))^;
  FH5T_STD_B32BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B32BE_g'))^;
  FH5T_STD_B32LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B32LE_g'))^;
  FH5T_STD_B64BE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B64BE_g'))^;
  FH5T_STD_B64LE := Phid_t(GetDllProc(FHandle, 'H5T_STD_B64LE_g'))^;
  FH5T_STD_REF_OBJ := Phid_t(GetDllProc(FHandle, 'H5T_STD_REF_OBJ_g'))^;
  FH5T_STD_REF_DSETREG := Phid_t(GetDllProc(FHandle, 'H5T_STD_REF_DSETREG_g'))^;
  FH5T_UNIX_D32BE := Phid_t(GetDllProc(FHandle, 'H5T_UNIX_D32BE_g'))^;
  FH5T_UNIX_D32LE := Phid_t(GetDllProc(FHandle, 'H5T_UNIX_D32LE_g'))^;
  FH5T_UNIX_D64BE := Phid_t(GetDllProc(FHandle, 'H5T_UNIX_D64BE_g'))^;
  FH5T_UNIX_D64LE := Phid_t(GetDllProc(FHandle, 'H5T_UNIX_D64LE_g'))^;
  FH5T_C_S1 := Phid_t(GetDllProc(FHandle, 'H5T_C_S1_g'))^;
  FH5T_FORTRAN_S1 := Phid_t(GetDllProc(FHandle, 'H5T_FORTRAN_S1_g'))^;
  FH5T_VAX_F32 := Phid_t(GetDllProc(FHandle, 'H5T_VAX_F32_g'))^;
  FH5T_VAX_F64 := Phid_t(GetDllProc(FHandle, 'H5T_VAX_F64_g'))^;
  FH5T_NATIVE_SCHAR := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_SCHAR_g'))^;
  FH5T_NATIVE_UCHAR := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UCHAR_g'))^;
  FH5T_NATIVE_SHORT := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_SHORT_g'))^;
  FH5T_NATIVE_USHORT := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_USHORT_g'))^;
  FH5T_NATIVE_INT := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_g'))^;
  FH5T_NATIVE_UINT := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_g'))^;
  FH5T_NATIVE_LONG := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_LONG_g'))^;
  FH5T_NATIVE_ULONG := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_ULONG_g'))^;
  FH5T_NATIVE_LLONG := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_LLONG_g'))^;
  FH5T_NATIVE_ULLONG := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_ULLONG_g'))^;
  FH5T_NATIVE_FLOAT := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_FLOAT_g'))^;
  FH5T_NATIVE_DOUBLE := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_DOUBLE_g'))^;
  FH5T_NATIVE_B8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_B8_g'))^;
  FH5T_NATIVE_B16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_B16_g'))^;
  FH5T_NATIVE_B32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_B32_g'))^;
  FH5T_NATIVE_B64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_B64_g'))^;
  FH5T_NATIVE_OPAQUE := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_OPAQUE_g'))^;
  FH5T_NATIVE_HADDR := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_HADDR_g'))^;
  FH5T_NATIVE_HSIZE := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_HSIZE_g'))^;
  FH5T_NATIVE_HSSIZE := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_HSSIZE_g'))^;
  FH5T_NATIVE_HERR := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_HERR_g'))^;
  FH5T_NATIVE_HBOOL := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_HBOOL_g'))^;
  FH5T_NATIVE_INT8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT8_g'))^;
  FH5T_NATIVE_UINT8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT8_g'))^;
  FH5T_NATIVE_INT_LEAST8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_LEAST8_g'))^;
  FH5T_NATIVE_UINT_LEAST8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_LEAST8_g'))^;
  FH5T_NATIVE_INT_FAST8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_FAST8_g'))^;
  FH5T_NATIVE_UINT_FAST8 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_FAST8_g'))^;
  FH5T_NATIVE_INT16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT16_g'))^;
  FH5T_NATIVE_UINT16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT16_g'))^;
  FH5T_NATIVE_INT_LEAST16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_LEAST16_g'))^;
  FH5T_NATIVE_UINT_LEAST16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_LEAST16_g'))^;
  FH5T_NATIVE_INT_FAST16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_FAST16_g'))^;
  FH5T_NATIVE_UINT_FAST16 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_FAST16_g'))^;
  FH5T_NATIVE_INT32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT32_g'))^;
  FH5T_NATIVE_UINT32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT32_g'))^;
  FH5T_NATIVE_INT_LEAST32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_LEAST32_g'))^;
  FH5T_NATIVE_UINT_LEAST32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_LEAST32_g'))^;
  FH5T_NATIVE_INT_FAST32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_FAST32_g'))^;
  FH5T_NATIVE_UINT_FAST32 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_FAST32_g'))^;
  FH5T_NATIVE_INT64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT64_g'))^;
  FH5T_NATIVE_UINT64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT64_g'))^;
  FH5T_NATIVE_INT_LEAST64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_LEAST64_g'))^;
  FH5T_NATIVE_UINT_LEAST64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_LEAST64_g'))^;
  FH5T_NATIVE_INT_FAST64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_INT_FAST64_g'))^;
  FH5T_NATIVE_UINT_FAST64 := Phid_t(GetDllProc(FHandle, 'H5T_NATIVE_UINT_FAST64_g'))^;
  FH5E_ERR_CLS := Phid_t(GetDllProc(FHandle, 'H5E_ERR_CLS_g'))^;
  FH5P_CLS_ROOT_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_ROOT_ID_g'))^;
  FH5P_CLS_OBJECT_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_OBJECT_CREATE_ID_g'))^;
  FH5P_CLS_FILE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_FILE_CREATE_ID_g'))^;
  FH5P_CLS_FILE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_FILE_ACCESS_ID_g'))^;
  FH5P_CLS_DATASET_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_DATASET_CREATE_ID_g'))^;
  FH5P_CLS_DATASET_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_DATASET_ACCESS_ID_g'))^;
  FH5P_CLS_DATASET_XFER_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_DATASET_XFER_ID_g'))^;
  FH5P_CLS_FILE_MOUNT_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_FILE_MOUNT_ID_g'))^;
  FH5P_CLS_GROUP_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_GROUP_CREATE_ID_g'))^;
  FH5P_CLS_GROUP_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_GROUP_ACCESS_ID_g'))^;
  FH5P_CLS_DATATYPE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_DATATYPE_CREATE_ID_g'))^;
  FH5P_CLS_DATATYPE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_DATATYPE_ACCESS_ID_g'))^;
  FH5P_CLS_STRING_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_STRING_CREATE_ID_g'))^;
  FH5P_CLS_ATTRIBUTE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_ATTRIBUTE_CREATE_ID_g'))^;
  FH5P_CLS_ATTRIBUTE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_ATTRIBUTE_ACCESS_ID_g'))^;
  FH5P_CLS_OBJECT_COPY_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_OBJECT_COPY_ID_g'))^;
  FH5P_CLS_LINK_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_LINK_CREATE_ID_g'))^;
  FH5P_CLS_LINK_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_CLS_LINK_ACCESS_ID_g'))^;
  FH5P_LST_FILE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_FILE_CREATE_ID_g'))^;
  FH5P_LST_FILE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_FILE_ACCESS_ID_g'))^;
  FH5P_LST_DATASET_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_DATASET_CREATE_ID_g'))^;
  FH5P_LST_DATASET_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_DATASET_ACCESS_ID_g'))^;
  FH5P_LST_DATASET_XFER_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_DATASET_XFER_ID_g'))^;
  FH5P_LST_FILE_MOUNT_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_FILE_MOUNT_ID_g'))^;
  FH5P_LST_GROUP_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_GROUP_CREATE_ID_g'))^;
  FH5P_LST_GROUP_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_GROUP_ACCESS_ID_g'))^;
  FH5P_LST_DATATYPE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_DATATYPE_CREATE_ID_g'))^;
  FH5P_LST_DATATYPE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_DATATYPE_ACCESS_ID_g'))^;
  FH5P_LST_ATTRIBUTE_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_ATTRIBUTE_CREATE_ID_g'))^;
  FH5P_LST_ATTRIBUTE_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_ATTRIBUTE_ACCESS_ID_g'))^;
  FH5P_LST_OBJECT_COPY_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_OBJECT_COPY_ID_g'))^;
  FH5P_LST_LINK_CREATE_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_LINK_CREATE_ID_g'))^;
  FH5P_LST_LINK_ACCESS_ID := Phid_t(GetDllProc(FHandle, 'H5P_LST_LINK_ACCESS_ID_g'))^;
end;

destructor THDF5Dll.Destroy;
begin
  if FHandle <> 0 then
    FreeLibrary(FHandle);
  inherited;
end;

function THDF5Dll.IsValid: Boolean;
begin
  Result := (FHandle <> 0);
end;

end.

