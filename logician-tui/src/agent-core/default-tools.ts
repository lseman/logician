import type { Tool } from "./types.ts";
import { bash } from "./tools/bash.ts";
import { edit_file } from "./tools/edit-file.ts";
import { file_diff } from "./tools/file-diff.ts";
import { find } from "./tools/find.ts";
import { git } from "./tools/git.ts";
import { list_files } from "./tools/list-files.ts";
import { read_file } from "./tools/read-file.ts";
import { rg_search } from "./tools/search.ts";
import { todo_write } from "./tools/todo-write.ts";
import { write_file } from "./tools/write-file.ts";

export function createDefaultTools(): Tool[] {
    return [
        list_files,
        find,
        read_file,
        rg_search,
        edit_file,
        write_file,
        file_diff,
        bash,
        git,
        todo_write,
    ];
}
